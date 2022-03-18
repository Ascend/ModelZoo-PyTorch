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

import os
import cv2
from PIL import Image
import numpy as np
import torch
from albumentations import Normalize, Resize, LongestMaxSize, PadIfNeeded
from albumentations.pytorch import ToTensorV2 as ToTensor
from albumentations import Compose
from tqdm import tqdm
import argparse

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

if __name__ == '__main__':
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

    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=str, default='/opt/npu/VOCdevkit/VOC2012')
    parser.add_argument('--bin-dir', type=str, default='./prepare_dataset')
    args = parser.parse_args()
    i = 0
    with open(os.path.join(args.root_dir, 'ImageSets', 'Segmentation', 'val.txt'), 'r') as f:
        file_names = [x.strip() for x in f.readlines()]
        print(len(file_names))
        for file in tqdm(file_names):
            input_image = Image.open(os.path.join(args.root_dir, 'JPEGImages', f"{file}.jpg")).convert('RGB')
            input_target = Image.open(os.path.join(args.root_dir, 'SegmentationClass', f"{file}.png"))
            input_tensor, target_tensor = val_transforms(input_image, input_target)
            img = np.array(input_tensor).astype(np.float32)
            label = np.array(target_tensor).astype(np.float32)
            img.tofile(os.path.join(args.bin_dir, file + ".bin"))