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

import math

import numpy as np
from PIL import Image
import torch
import torch.utils.data
from torchvision import transforms

DEFAULT_CROP_PCT = 0.875
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def _pil_interp(method):
    if method == 'bicubic':
        return transforms.InterpolationMode.BICUBIC
    elif method == 'lanczos':
        return transforms.InterpolationMode.LANCZOS
    elif method == 'hamming':
        return transforms.InterpolationMode.HAMMING
    else:
        return transforms.InterpolationMode.BILINEAR


def transforms_imagenet_eval(
        img_size=224,
        crop_pct=None,
        interpolation='bilinear',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD):
    crop_pct = crop_pct or DEFAULT_CROP_PCT

    if isinstance(img_size, tuple):
        assert len(img_size) == 2
        if img_size[-1] == img_size[-2]:
            # fall-back to older behaviour so Resize scales to shortest edge if target is square
            scale_size = int(math.floor(img_size[0] / crop_pct))
        else:
            scale_size = tuple([int(x / crop_pct) for x in img_size])
    else:
        scale_size = int(math.floor(img_size / crop_pct))

    tfl = [
        transforms.Resize(scale_size, _pil_interp(interpolation)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
                    mean=torch.tensor(mean),
                    std=torch.tensor(std))
    ]

    return transforms.Compose(tfl)


def create_loader(
        dataset,
        input_size,
        batch_size,
        interpolation='bilinear',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        num_workers=1,
        crop_pct=None
):
    if isinstance(input_size, tuple):
        img_size = input_size[-2:]
    else:
        img_size = input_size

    transform = transforms_imagenet_eval(
        img_size,
        interpolation=interpolation,
        mean=mean,
        std=std,
        crop_pct=crop_pct)

    dataset.transform = transform

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=torch.utils.data.dataloader.default_collate,
        drop_last=True,
    )

    return loader
