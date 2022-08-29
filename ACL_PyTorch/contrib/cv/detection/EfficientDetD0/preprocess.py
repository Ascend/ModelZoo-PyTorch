# Copyright 2021 Huawei Technologies Co., Ltd
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
sys.path.append(r'./efficientdet-pytorch')
import os
import argparse
from effdet import create_dataset, create_loader
from effdet.data import resolve_input_config
from timm.utils import *

parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')

parser.add_argument('--root', default='', type=str, metavar='DIR',
                    help='path to dataset root')
parser.add_argument('--dataset', default='coco', type=str, metavar='DATASET',
                    help='Name of dataset (default: "coco"')
parser.add_argument('--split', default='val',
                    help='validation split')
parser.add_argument('--model', '-m', metavar='MODEL', default='tf_efficientdet_d0',
                    help='model architecture (default: tf_efficientdet_d1)')
parser.add_argument('--bin-save', default='', type=str, metavar='save',
                    help='path to save bin')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')

args = parser.parse_args()
setup_default_logging()
dataset = create_dataset(args.dataset, args.root, args.split)
if args.model == 'tf_efficientdet_d0':
    model_config = {'input_size': (3, 512, 512),
                    'interpolation': 'bilinear',
                    'mean': (0.485, 0.456, 0.406),
                    'std': (0.229, 0.224, 0.225),
                    'fill_color': 'mean'}
elif args.model == 'tf_efficientdet_d7':
    model_config = {'input_size': (3, 1536, 1536),
                    'interpolation': 'bilinear',
                    'mean': (0.485, 0.456, 0.406),
                    'std': (0.229, 0.224, 0.225),
                    'fill_color': 'mean'}
input_config = resolve_input_config(args, model_config)
print(args)
loader = create_loader(
    dataset,
    input_size=input_config['input_size'],
    batch_size=args.batch_size,
    use_prefetcher=True,
    interpolation=input_config['interpolation'],
    fill_color=input_config['fill_color'],
    mean=input_config['mean'],
    std=input_config['std'],
    num_workers=4,
    pin_mem=True,
)
pic=os.listdir(os.path.join(args.root,'val2017'))
pic.sort()

if not os.path.exists(args.bin_save):
    os.makedirs(args.bin_save)

for i, file in zip(loader, pic):
    img = i[0].numpy()
    print(file)
    img.tofile(os.path.join(args.bin_save, file.split('.')[0] + ".bin"))