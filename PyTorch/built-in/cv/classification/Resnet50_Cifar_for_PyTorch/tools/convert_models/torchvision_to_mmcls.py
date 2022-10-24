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
from pathlib import Path

import torch


def convert_resnet(src_dict, dst_dict):
    """convert resnet checkpoints from torchvision."""
    for key, value in src_dict.items():
        if not key.startswith('fc'):
            dst_dict['backbone.' + key] = value
        else:
            dst_dict['head.' + key] = value


# model name to convert function
CONVERT_F_DICT = {
    'resnet': convert_resnet,
}


def convert(src: str, dst: str, convert_f: callable):
    print('Converting...')
    blobs = torch.load(src, map_location='cpu')
    converted_state_dict = OrderedDict()

    # convert key in weight
    convert_f(blobs, converted_state_dict)

    torch.save(converted_state_dict, dst)
    print('Done!')


def main():
    parser = argparse.ArgumentParser(description='Convert model keys')
    parser.add_argument('src', help='src detectron model path')
    parser.add_argument('dst', help='save path')
    parser.add_argument(
        'model', type=str, help='The algorithm needs to change the keys.')
    args = parser.parse_args()

    dst = Path(args.dst)
    if dst.suffix != '.pth':
        print('The path should contain the name of the pth format file.')
        exit(1)
    dst.parent.mkdir(parents=True, exist_ok=True)

    # this tool only support model in CONVERT_F_DICT
    support_models = list(CONVERT_F_DICT.keys())
    if args.model not in CONVERT_F_DICT:
        print(f'The "{args.model}" has not been supported to convert now.')
        print(f'This tool only supports {", ".join(support_models)}.')
        print('If you have done the converting job, PR is welcome!')
        exit(1)

    convert_f = CONVERT_F_DICT[args.model]
    convert(args.src, args.dst, convert_f)


if __name__ == '__main__':
    main()
