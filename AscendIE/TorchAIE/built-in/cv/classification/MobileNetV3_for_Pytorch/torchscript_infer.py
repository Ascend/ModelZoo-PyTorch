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

# /home/devkit1/liulanxi/ModelZoo-PyTorch/ACL_PyTorch/built-in/cv/MobileNetV3_for_Pytorch/mobilenetv3/mobileNetV3.compiled.ts
import argparse
import time
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch_aie
from torch_aie import _enums
from torch.nn import functional as F
import numpy as np

from mobilenetv3 import MobileNetV3_Small
from data import Dataset, create_loader, compute_accuracy, AverageMeter
from torchscript_export import torchscript_export

def main(args):
    BATCH_SIZE = args.batch_size
    ts_path = torchscript_export(BATCH_SIZE)
    ts_model = torch.jit.load(ts_path)
    torch_aie.set_device(0)
    try:
        compiled_module = torch_aie.compile(
        ts_model,
        inputs= [torch_aie.Input((BATCH_SIZE, 3, 224, 224))],
        precision_policy=_enums.PrecisionPolicy.FP16,
        soc_version="Ascend310P3"
        )
        print("MobileNetV3 model compiled successfully.")

    except Exception as e:
        print(f"Compilation erorr:{e}")
        exit(1)
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
    model = compiled_module
    # infer and compute accuracy
    top1 = AverageMeter()
    top5 = AverageMeter()
    inference_time = []
    for i, (input_data, target) in enumerate(tqdm(loader)):
        input_npu = input_data.to("npu:0")
        start_time = time.time()
        output_npu = model.forward(input_npu)
        end_time = time.time()
        output = output_npu.to("cpu")
        if i >= 5:
            inference_time.append(end_time - start_time)
        prec1, prec5 = compute_accuracy(output, target, topk=(1, 5))
        top1.update(prec1.item(), input_data.size(0))
        top5.update(prec5.item(), input_data.size(0))
    print(f'batch_size = {BATCH_SIZE}')
    print(f'ACC: Top1@ {top1.avg:.3f} | Top5@ {top5.avg:.3f}')
    avg_infer_time = sum(inference_time) / len(inference_time)
    print(f'FPS = {BATCH_SIZE / avg_infer_time}')    


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
    parser.add_argument('--workers', default=32, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('--device_id', default=0, type=int, help='device id')

    args = parser.parse_args()
    main(args)
