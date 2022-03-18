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
import re
from collections import OrderedDict

import torch


def convert(in_file, out_file):
    """Convert keys in checkpoints.

    There can be some breaking changes during the development of mmdetection,
    and this tool is used for upgrading checkpoints trained with old versions
    to the latest one.
    """
    checkpoint = torch.load(in_file)
    in_state_dict = checkpoint.pop('state_dict')
    out_state_dict = OrderedDict()
    for key, val in in_state_dict.items():
        # Use ConvModule instead of nn.Conv2d in RetinaNet
        # cls_convs.0.weight -> cls_convs.0.conv.weight
        m = re.search(r'(cls_convs|reg_convs).\d.(weight|bias)', key)
        if m is not None:
            param = m.groups()[1]
            new_key = key.replace(param, 'conv.{}'.format(param))
            out_state_dict[new_key] = val
            continue

        out_state_dict[key] = val
    checkpoint['state_dict'] = out_state_dict
    torch.save(checkpoint, out_file)


def main():
    parser = argparse.ArgumentParser(description='Upgrade model version')
    parser.add_argument('in_file', help='input checkpoint file')
    parser.add_argument('out_file', help='output checkpoint file')
    args = parser.parse_args()
    convert(args.in_file, args.out_file)


if __name__ == '__main__':
    main()
