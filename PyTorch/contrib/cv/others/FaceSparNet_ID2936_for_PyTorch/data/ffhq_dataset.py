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

import os
import random
import numpy as np
from PIL import Image
import imgaug as ia
import imgaug.augmenters as iaa

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from data.base_dataset import BaseDataset


class FFHQDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.img_size = opt.load_size
        self.shuffle = True if opt.isTrain else False 

        self.img_dir = opt.dataroot
        self.img_names = self.get_img_names()

        self.to_tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

    def get_img_names(self,):
        img_names = [x for x in os.listdir(self.img_dir)] 
        if self.shuffle:
            random.shuffle(img_names)
        return img_names

    def __len__(self,):
        return len(self.img_names)

    def __getitem__(self, idx):
        sample = {}
        img_path = os.path.join(self.img_dir, self.img_names[idx])

        hr_img = Image.open(img_path).convert('RGB')
        hr_img = hr_img.resize((self.img_size, self.img_size))
        hr_img = random_gray(hr_img, p=0.3)
        scale_size = np.random.randint(32, 128)
        lr_img = complex_imgaug(hr_img, self.img_size, scale_size)

        hr_tensor = self.to_tensor(hr_img)
        lr_tensor = self.to_tensor(lr_img)

        return {'HR': hr_tensor, 'LR': lr_tensor, 'HR_paths': img_path}


def complex_imgaug(x, org_size, scale_size):
    """input single RGB PIL Image instance"""
    x = np.array(x)
    x = x[np.newaxis, :, :, :]
    aug_seq = iaa.Sequential([
            iaa.Sometimes(0.5, iaa.OneOf([
                iaa.GaussianBlur((3, 15)),
                iaa.AverageBlur(k=(3, 15)),
                iaa.MedianBlur(k=(3, 15)),
                iaa.MotionBlur((5, 25))
            ])),
            iaa.Resize(scale_size, interpolation=ia.ALL),
            iaa.Sometimes(0.2, iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.1*255), per_channel=0.5)),
            iaa.Sometimes(0.7, iaa.JpegCompression(compression=(10, 65))),
            iaa.Resize(org_size),
        ])
    
    aug_img = aug_seq(images=x)
    return aug_img[0]


def random_gray(x, p=0.5):
    """input single RGB PIL Image instance"""
    x = np.array(x)
    x = x[np.newaxis, :, :, :]
    aug = iaa.Sometimes(p, iaa.Grayscale(alpha=1.0)) 
    aug_img = aug(images=x)
    return aug_img[0]

