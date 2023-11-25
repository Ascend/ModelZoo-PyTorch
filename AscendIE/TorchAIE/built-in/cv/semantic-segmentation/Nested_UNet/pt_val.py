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

import os
import yaml
import json
import cv2
import argparse
import numpy as np
import torch
import torch_aie

from torch_aie import _enums
from torch.utils.data import dataloader
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose

from model_pt import forward_nms_script

WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))  # DPP


class InfiniteDataLoader(dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler:
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


def collate_fn(batch):
    img = batch  # transposed
    return img


def create_dataloader(data_path, batch_size, workers=8):
    dataset = []
    file = open("./pytorch-nested-unet/val_ids.txt")
    val_ids = file.read().split('\n')

    val_transform = Compose([
        transforms.Resize(96, 96),
        transforms.Normalize(),
    ])
    multi = opt.multi
    for i in range(multi):
        print(i / multi)
        val_len = len(val_ids)
        val_ids_tail = val_ids[-1]
        while (val_len % batch_size != 0):
            val_ids.append(val_ids_tail)
            val_len += 1
        for img_id in val_ids:
            if len(img_id) == 0: continue
            img = cv2.imread(os.path.join("./pytorch-nested-unet/inputs/dsb2018_96/images", img_id + '.png'))
            augmented = val_transform(image=img)
            img = augmented['image']
            img = img.astype('float32') / 255
            img = img.transpose(2, 0, 1)
            dataset.append(img)

    batch_size = min(batch_size, len(dataset))
    loader = InfiniteDataLoader  # only DataLoader allows for attribute updates
    nw = min([os.cpu_count() // WORLD_SIZE, batch_size if batch_size > 1 else 0, workers])  # number of workers
    return loader(dataset,
                  batch_size=batch_size,
                  shuffle=False,
                  num_workers=nw,
                  sampler=None,
                  pin_memory=True,
                  collate_fn=collate_fn), dataset


def main(opt):
    # load model
    model = torch.jit.load(opt.model)
    print("Loading model: ", opt.model)
    torch_aie.set_device(opt.device_id)
    if opt.need_compile:
        inputs = []
        inputs.append(torch_aie.Input((opt.batch_size, 3, 96, 96)))
        model = torch_aie.compile(
            model,
            inputs=inputs,
            precision_policy=_enums.PrecisionPolicy.FP16,
            truncate_long_and_double=True,
            require_full_compilation=False,
            allow_tensor_replace_int=False,
            min_block_size=3,
            torch_executed_ops=[],
            soc_version=opt.soc_version,
            optimization_level=0)

    # load dataset
    dataloader = create_dataloader(opt.data_path, opt.batch_size)[0]
    # inference & nms
    pred_results = forward_nms_script(model, dataloader, opt.batch_size, opt.device_id)
    if opt.multi == 1:
        file = open(opt.val_ids_file)
        val_ids = file.read().split('\n')
        for index, input_fname in enumerate(val_ids):
            result_fname = input_fname + '.bin'
            result_path = os.path.join(opt.result_root_path, f"result_bs{opt.batch_size}")
            result_file = os.path.join(result_path, result_fname)
            if (os.path.exists(result_path) == False):
                os.makedirs(result_path)
            np.array(pred_results[index // opt.batch_size][index % opt.batch_size].numpy().tofile(result_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Nested_UNet offline model inference.')
    parser.add_argument('--data_path', type=str, default="prep_data", help='root dir for val images and annotations')
    parser.add_argument('--result_root_path', type=str, default="result", help='root dir for result')
    parser.add_argument('--val_ids_file', type=str, default="./pytorch-nested-unet/val_ids.txt",
                        help='path for val_ids.txt')
    parser.add_argument('--soc_version', type=str, default='Ascend310P3', help='soc version')
    parser.add_argument('--model', type=str, default="nested_unet_torch_aie.pt", help='ts model path')
    parser.add_argument('--need_compile', action="store_true", help='if the loaded model needs to be compiled or not')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--device_id', type=int, default=0, help='device id')
    parser.add_argument('--multi', type=int, default=1,
                        help='multiples of dataset replication for enough infer loop. if multi != 1, the pred result will not be stored.')

    opt = parser.parse_args()

    main(opt)
