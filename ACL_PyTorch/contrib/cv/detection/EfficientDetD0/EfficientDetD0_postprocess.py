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
import numpy as np
import time
import os
import argparse
import torch
from effdet import create_evaluator, create_dataset, create_loader, create_model
from effdet.data import resolve_input_config
from timm.utils import *
from timm.models.layers import set_layer_config
from effdet.bench import DetBenchPredict
from effdet.config import get_efficientdet_config
from tqdm import tqdm

parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('--root', default='', type=str, metavar='DIR',
                    help='path to dataset root')
parser.add_argument('--omfile', type=str,
                    help='om inference bin file save path')
parser.add_argument('--dataset', default='coco', type=str, metavar='DATASET',
                    help='Name of dataset (default: "coco"')
parser.add_argument('--split', default='val',
                    help='validation split')
parser.add_argument('--model', '-m', metavar='MODEL', default='tf_efficientdet_d0',
                    help='model architecture (default: tf_efficientdet_d1)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')

if __name__ == '__main__':
    args = parser.parse_args()
    setup_default_logging()
    config = get_efficientdet_config(model_name='tf_efficientdet_d0')
    with set_layer_config(scriptable=False):
        extra_args = {}
        bench = DetBenchPredict(config)

    dataset = create_dataset(args.dataset, args.root, args.split)
    model_config = bench.config
    param_count = sum([m.numel() for m in bench.parameters()])
    input_config = resolve_input_config(args, model_config)
    loader = create_loader(
        dataset,
        input_size=input_config['input_size'],
        batch_size=1,
        use_prefetcher=True,
        interpolation=input_config['interpolation'],
        fill_color=input_config['fill_color'],
        mean=input_config['mean'],
        std=input_config['std'],
        num_workers=4,
        pin_mem=True,
    )
    evaluator = create_evaluator(args.dataset, dataset, pred_yxyx=False)
    bench.eval()
    batch_time = AverageMeter()
    end = time.time()
    last_idx = len(loader) - 1

    om_data = args.omfile
    om_files = list(os.listdir(om_data))
    files = list(set([file.split('_')[0] for file in om_files]))
    files.sort()

    box_list = [i for i in range(5, 10)]
    class_list = [i for i in range(0, 5)]
    with torch.no_grad():
        for (i, (input, target)), file in zip(enumerate(loader), tqdm(files)):
            size = 128
            box_out, class_out = [], []
            for box, class_ in zip(box_list, class_list):
                size /= 2
                box_file = om_data + '/' + str(file) + "_"+ str(box) + '.bin'
                class_file = om_data + '/' + str(file) + "_"+ str(class_) + '.bin'
                box_data = np.fromfile(box_file, dtype=np.float32)
                class_data = np.fromfile(class_file, dtype=np.float32)
                box_data.shape = 1, 36, int(size), int(size)
                class_data.shape = 1, 810, int(size), int(size)
                box_data = torch.from_numpy(box_data)
                class_data = torch.from_numpy(class_data)
                box_out.append(box_data)
                class_out.append(class_data)
            output = bench(x=input, class_out=class_out, box_out=box_out, img_info=target)
            evaluator.add_predictions(output, target)
            batch_time.update(time.time() - end)
            end = time.time()
    mean_ap = 0.
    if dataset.parser.has_labels:
        mean_ap = evaluator.evaluate()