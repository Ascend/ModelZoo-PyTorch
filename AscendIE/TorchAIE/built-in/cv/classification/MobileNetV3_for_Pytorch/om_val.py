# Copyright 2022 Huawei Technologies Co., Ltd
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


import argparse
from tqdm import tqdm
from collections import OrderedDict

import torch
from torch.nn import functional as F
import numpy as np
from ais_bench.infer.interface import InferSession

from mobilenetv3 import MobileNetV3_Small
from data import Dataset, create_loader, compute_accuracy, AverageMeter


def adjust_checkpoint(checkpoint):
    new_state_dict = OrderedDict()
    for key, value in checkpoint.items():
        if key == "module.features.0.0.weight":
            print(value)
        if key[0:7] == "module.":
            name = key[7:]
        else:
            name = key[0:]
        
        new_state_dict[name] = value
    return new_state_dict


def main(args):
    # create model
    flag = args.checkpoint.split('.')[1]
    if flag == 'pth':
        checkpoint = torch.load(args.checkpoint, map_location='cpu')['state_dict']
        checkpoint = adjust_checkpoint(checkpoint)
        model = MobileNetV3_Small()
        model.load_state_dict(checkpoint)
        model.eval()
    elif flag == 'om':
        model = InferSession(args.device_id, args.checkpoint)

    # create dataloader
    loader = create_loader(
        Dataset(args.dataset_dir),
        input_size=(3, args.img_size, args.img_size),
        batch_size=args.batch_size,
        interpolation=args.interpolation,
        mean=args.mean,
        std=args.std,
        num_workers=args.workers,
        crop_pct=args.crop_pct)

    # infer and compute accuracy
    top1 = AverageMeter()
    top5 = AverageMeter()
    for i, (input_data, target) in enumerate(tqdm(loader)):
        if flag == 'pth':
            output = model(input_data)
        elif flag == 'om':
            if i == len(loader) - 1:
                input_data = F.pad(input_data, (0, 0, 0, 0, 0, 0, 0, args.batch_size-len(target)), "constant", 0)
            output = model.infer([input_data.numpy().astype(np.float16)])[0]
            if i == len(loader) - 1:
                output = output[:len(target)]
            output = torch.Tensor(output)

        # measure accuracy and record loss
        prec1, prec5 = compute_accuracy(output, target, topk=(1, 5))
        top1.update(prec1.item(), input_data.size(0))
        top5.update(prec5.item(), input_data.size(0))

    print(f'ACC: Top1@ {top1.avg:.3f} | Top5@ {top5.avg:.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
    parser.add_argument('--checkpoint', default='output/mbv3_small_bs32.om', type=str, metavar='PATH',
                        help='path to latest checkpoint')
    parser.add_argument('--dataset_dir', default='imagenet', type=str, metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--batch-size', default=1, type=int,
                        metavar='N', help='mini-batch size')
    parser.add_argument('--img-size', default=224, type=int,
                        metavar='N', help='Input image dimension')
    parser.add_argument('--mean', default=(0.485, 0.456, 0.406), type=float, nargs='+', metavar='MEAN',
                        help='Override mean pixel value of dataset')
    parser.add_argument('--std', default=(0.229, 0.224, 0.225), type=float, nargs='+',  metavar='STD',
                        help='Override std deviation of of dataset')
    parser.add_argument('--crop-pct', default=0.875, type=float, metavar='PCT',
                        help='Override default crop pct of 0.875')
    parser.add_argument('--interpolation', default='bilinear', type=str, metavar='NAME',
                        help='Image resize interpolation type (overrides model)')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('--device_id', default=0, type=int, help='device id')

    args = parser.parse_args()
    main(args)
