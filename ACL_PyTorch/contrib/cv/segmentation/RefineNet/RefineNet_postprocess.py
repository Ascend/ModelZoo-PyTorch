# Copyright 2020 Huawei Technologies Co., Ltd
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

import torch
import cv2
import torch.nn.functional as F
import densetorch as dt
import numpy as np
import os
import argparse

from torchvision.datasets.voc import VOCSegmentation
from torch.utils.data import DataLoader
from tqdm import tqdm
from albumentations import Normalize
from albumentations.pytorch import ToTensorV2 as ToTensor
from albumentations import Compose, PadIfNeeded, LongestMaxSize
from multiprocessing import Pool

class Alb_Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        keys = ["image", "mask"]
        np_dtypes = [np.float32, np.uint8]
        torch_dtypes = [torch.float32, torch.long]
        sample_dict = {
            key: np.array(value, dtype=dtype)
            for key, value, dtype in zip(keys, [image, target], np_dtypes)
        }
        output = Compose(self.transforms )(**sample_dict)
        return [output[key].to(dtype) for key, dtype in zip(keys, torch_dtypes)]

def setup_data_loaders(root):
    wrapper = Alb_Compose
    common_transformations = [
        Normalize(max_pixel_value=255, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor(),
    ]

    val_transforms = wrapper([LongestMaxSize(max_size=500),
                              PadIfNeeded(
                                    min_height=500,
                                    min_width=500,
                                    border_mode=cv2.BORDER_CONSTANT,
                                    value=np.array((0.485, 0.456, 0.406)) * 255,
                                    mask_value=255,
                              )] + common_transformations)

    val_set = VOCSegmentation(
        root=root,
        image_set="val",
        year="2012",
        download=0,
        transforms=val_transforms,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=False,
        drop_last=False,
    )
    return val_loader

def maybe_cast_target_to_long(target):
    """Torch losses usually work on Long types"""
    if target.dtype == torch.uint8:
        return target.to(torch.long)
    return target

def get_input_and_targets(sample, dataloader, device):
    if isinstance(sample, dict):
        input = sample["image"].float().to(device)
        targets = [
            maybe_cast_target_to_long(sample[k].to(device))
            for k in dataloader.dataset.masks_names
        ]
    elif isinstance(sample, (tuple, list)):
        input, *targets = sample
        input = input.float().to(device)
        targets = [maybe_cast_target_to_long(target.to(device)) for target in targets]
    else:
        raise Exception(f"Sample type {type(sample)} is not supported.")
    return input, targets

def get_val(metrics):
    results = [(m.name, m.val()) for m in metrics]
    names, vals = list(zip(*results))
    out = ["{} : {:4f}".format(name, val) for name, val in results]
    return vals, " | ".join(out)

def task(idx, file_names):
    start = idx * 150
    end = min((idx + 1) * 150, len(file_names))
    # print(f'subprocess {idx} running. start: {start}, end: {end}')
    outputs_sub = []
    for i in range(start, end):
        file = file_names[i]
        with open(os.path.join(args.result_dir, file + '_0.txt')) as res_f:
            output = []
            for line in res_f:
                num_list = line.split()
                for num in num_list:
                    output.append(float(num))
            output = torch.from_numpy(np.array(output).reshape((1, 21, 125, 125)))
            outputs_sub.append(output)
    # print(f'subprocess {idx} done. outputs_sub length: {len(outputs_sub)}')
    return outputs_sub

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--val-dir', type=str, default='/opt/npu/')
    parser.add_argument('--result-dir', type=str, default='result/dumpOutput_device0')
    args = parser.parse_args()

    validation_loss = dt.engine.MeanIoU(num_classes=21)
    val_loader = setup_data_loaders(args.val_dir)
    metrics = dt.misc.utils.make_list(validation_loss)
    for metric in metrics:
        metric.reset()

    # 多进程读取outputs
    device = torch.device('cpu')
    root_dir = args.val_dir
    file_names = []
    outputs = []

    with open(os.path.join(root_dir, 'VOCdevkit', 'VOC2012', 'ImageSets', 'Segmentation', 'val.txt'), 'r') as f:
        file_names = [x.strip() for x in f.readlines()]
        pool = Pool(10)
        results = []
        for i in range(10):
            results.append(pool.apply_async(task, args=(i, file_names)))
        pool.close()
        pool.join()
        for res in results:
            outputs.extend(res.get())
        print(len(outputs))

    # 计算精度
    pbar = tqdm(val_loader)
    for idx, sample in enumerate(pbar):
        _, targets = get_input_and_targets(sample=sample, dataloader=val_loader, device=device)
        targets = [target.squeeze(dim=1).cpu().numpy() for target in targets]
        output = outputs[idx]
        output = dt.misc.utils.make_list(output)
        for out, target, metric in zip(output, targets, metrics):
            metric.update(
                F.interpolate(
                    out, size=target.shape[1:], mode="bilinear", align_corners=False
                )
                .squeeze(dim=1)
                .cpu()
                .numpy(),
                target,
            )

    print(f"Validation: ", get_val(metrics)[1])
