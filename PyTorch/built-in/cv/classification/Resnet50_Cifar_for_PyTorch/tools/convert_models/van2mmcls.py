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
import os.path as osp
from collections import OrderedDict

import mmcv
import torch
from mmcv.runner import CheckpointLoader


def convert_van(ckpt):

    new_ckpt = OrderedDict()

    for k, v in list(ckpt.items()):
        new_v = v
        if k.startswith('head'):
            new_k = k.replace('head.', 'head.fc.')
            new_ckpt[new_k] = new_v
            continue
        elif k.startswith('patch_embed'):
            if 'proj.' in k:
                new_k = k.replace('proj.', 'projection.')
            else:
                new_k = k
        elif k.startswith('block'):
            new_k = k.replace('block', 'blocks')
            if 'attn.spatial_gating_unit' in new_k:
                new_k = new_k.replace('conv0', 'DW_conv')
                new_k = new_k.replace('conv_spatial', 'DW_D_conv')
            if 'dwconv.dwconv' in new_k:
                new_k = new_k.replace('dwconv.dwconv', 'dwconv')
        else:
            new_k = k

        if not new_k.startswith('head'):
            new_k = 'backbone.' + new_k
        new_ckpt[new_k] = new_v
    return new_ckpt


def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in pretrained van models to mmcls style.')
    parser.add_argument('src', help='src model path or url')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()

    checkpoint = CheckpointLoader.load_checkpoint(args.src, map_location='cpu')

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    weight = convert_van(state_dict)
    mmcv.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(weight, args.dst)

    print('Done!!')


if __name__ == '__main__':
    main()
