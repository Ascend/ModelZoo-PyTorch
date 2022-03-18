# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from collections import OrderedDict

import mmcv
import torch

arch_settings = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3)}


def convert_bn(blobs, state_dict, caffe_name, torch_name, converted_names):
    # detectron replace bn with affine channel layer
    state_dict[torch_name + '.bias'] = torch.from_numpy(blobs[caffe_name +
                                                              '_b'])
    state_dict[torch_name + '.weight'] = torch.from_numpy(blobs[caffe_name +
                                                                '_s'])
    bn_size = state_dict[torch_name + '.weight'].size()
    state_dict[torch_name + '.running_mean'] = torch.zeros(bn_size)
    state_dict[torch_name + '.running_var'] = torch.ones(bn_size)
    converted_names.add(caffe_name + '_b')
    converted_names.add(caffe_name + '_s')


def convert_conv_fc(blobs, state_dict, caffe_name, torch_name,
                    converted_names):
    state_dict[torch_name + '.weight'] = torch.from_numpy(blobs[caffe_name +
                                                                '_w'])
    converted_names.add(caffe_name + '_w')
    if caffe_name + '_b' in blobs:
        state_dict[torch_name + '.bias'] = torch.from_numpy(blobs[caffe_name +
                                                                  '_b'])
        converted_names.add(caffe_name + '_b')


def convert(src, dst, depth):
    """Convert keys in detectron pretrained ResNet models to pytorch style."""
    # load arch_settings
    if depth not in arch_settings:
        raise ValueError('Only support ResNet-50 and ResNet-101 currently')
    block_nums = arch_settings[depth]
    # load caffe model
    caffe_model = mmcv.load(src, encoding='latin1')
    blobs = caffe_model['blobs'] if 'blobs' in caffe_model else caffe_model
    # convert to pytorch style
    state_dict = OrderedDict()
    converted_names = set()
    convert_conv_fc(blobs, state_dict, 'conv1', 'conv1', converted_names)
    convert_bn(blobs, state_dict, 'res_conv1_bn', 'bn1', converted_names)
    for i in range(1, len(block_nums) + 1):
        for j in range(block_nums[i - 1]):
            if j == 0:
                convert_conv_fc(blobs, state_dict,
                                'res{}_{}_branch1'.format(i + 1, j),
                                'layer{}.{}.downsample.0'.format(i, j),
                                converted_names)
                convert_bn(blobs, state_dict,
                           'res{}_{}_branch1_bn'.format(i + 1, j),
                           'layer{}.{}.downsample.1'.format(i, j),
                           converted_names)
            for k, letter in enumerate(['a', 'b', 'c']):
                convert_conv_fc(blobs, state_dict,
                                'res{}_{}_branch2{}'.format(i + 1, j, letter),
                                'layer{}.{}.conv{}'.format(i, j, k + 1),
                                converted_names)
                convert_bn(blobs, state_dict,
                           'res{}_{}_branch2{}_bn'.format(i + 1, j, letter),
                           'layer{}.{}.bn{}'.format(i, j,
                                                    k + 1), converted_names)
    # check if all layers are converted
    for key in blobs:
        if key not in converted_names:
            print('Not Convert: {}'.format(key))
    # save checkpoint
    checkpoint = dict()
    checkpoint['state_dict'] = state_dict
    torch.save(checkpoint, dst)


def main():
    parser = argparse.ArgumentParser(description='Convert model keys')
    parser.add_argument('src', help='src detectron model path')
    parser.add_argument('dst', help='save path')
    parser.add_argument('depth', type=int, help='ResNet model depth')
    args = parser.parse_args()
    convert(args.src, args.dst, args.depth)


if __name__ == '__main__':
    main()
