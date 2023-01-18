# Copyright (C) Huawei Technologies Co., Ltd 2022-2022. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
sys.path.append("../CSWin-Transformer")
import argparse
import linecache
import numpy as np
import yaml
import os
from timm.data import create_loader
from timm.utils import *
from labeled_memcached_dataset import McDataset
import warnings

warnings.filterwarnings('ignore')

config_parser = parser = argparse.ArgumentParser(description='Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='CSWin Training and Evaluating')

parser.add_argument('--data', default='/opt/npu/imagenet', metavar='DIR',
                    help='path to dataset')

parser.add_argument('--savepath', default='/home/Liu/savetxt', metavar='DIR',
                    help='path to save')


def _parse_args():
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    args = parser.parse_args(remaining)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def main():
    setup_default_logging()
    args, args_text = _parse_args()
    eval_dir = os.path.join(args.data, 'val')
    if not os.path.isdir(eval_dir):
        eval_dir = os.path.join(args.data, 'validation')
        if not os.path.isdir(eval_dir):
            exit(1)
    dataset_eval = McDataset(args.data, './dataset/ILSVRC2012_name_val.txt', 'val')

    loader_eval = create_loader(
        dataset_eval,
        input_size=(3, 224, 224),
        batch_size=1,
        is_training=False,
        use_prefetcher=False,
        interpolation='bicubic',
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        num_workers=8,
        distributed=False,
        crop_pct=0.875,
        pin_memory=False,
    )

    save_path = args.savepath

    f = open('./dataset/ILSVRC2012_name_val.txt')
    f.readlines()
    for batch_idx, (input, target) in enumerate(loader_eval):
        print(batch_idx)
        input1 = input.cpu().numpy()
        img = np.array(input1).astype(np.float32)
        output_path = os.path.join(save_path, linecache.getline(r'./dataset/ILSVRC2012_name_val.txt', \
                                                                batch_idx+1).split('/')[1].split('.')[0] + ".bin")
        img.tofile(output_path)


if __name__ == '__main__':
    main()
