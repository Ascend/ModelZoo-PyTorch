# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================

import os
import random
from random import shuffle
import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler,Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True
class ImageFolder(Dataset):
    def __init__(self, root,image_size=224,mode='train',augmentation_prob=0.4):
        """Initializes image paths and preprocessing module."""
        self.root = root
        
        # GT : Ground Truth
        self.GT_paths = root+'_GT/'
        self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        self.image_size = image_size
        self.mode = mode
        self.RotationDegree = [0,90,180,270]
        self.augmentation_prob = augmentation_prob
        print("image count in {} path :{}".format(self.mode,len(self.image_paths)))

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image_path = self.image_paths[index]
        filename = image_path.split('_')[-1][:-len(".jpg")]
        GT_path = self.GT_paths + 'ISIC_' + filename + '_segmentation.png'

        image = Image.open(image_path)
        GT = Image.open(GT_path)

        aspect_ratio = image.size[1]/image.size[0]

        Transform = []

        ResizeRange = random.randint(300,320)
        Transform.append(T.Resize((224,224)))
        p_transform = random.random()

        if (self.mode == 'train') and p_transform <= self.augmentation_prob:
            RotationDegree = random.randint(0,3)
            RotationDegree = self.RotationDegree[RotationDegree]
            if (RotationDegree == 90) or (RotationDegree == 270):
                aspect_ratio = 1/aspect_ratio

            Transform.append(T.RandomRotation((RotationDegree,RotationDegree)))
                        
            RotationRange = random.randint(-10,10)
            Transform.append(T.RandomRotation((RotationRange,RotationRange)))
            CropRange = random.randint(250,270)
            Transform.append(T.CenterCrop((int(CropRange*aspect_ratio),CropRange)))
            Transform = T.Compose(Transform)
            
            image = Transform(image)
            GT = Transform(GT)

            ShiftRange_left = random.randint(0,20)
            ShiftRange_upper = random.randint(0,20)
            ShiftRange_right = image.size[0] - random.randint(0,20)
            ShiftRange_lower = image.size[1] - random.randint(0,20)
            image = image.crop(box=(ShiftRange_left,ShiftRange_upper,ShiftRange_right,ShiftRange_lower))
            GT = GT.crop(box=(ShiftRange_left,ShiftRange_upper,ShiftRange_right,ShiftRange_lower))

            if random.random() < 0.5:
                image = F.hflip(image)
                GT = F.hflip(GT)

            if random.random() < 0.5:
                image = F.vflip(image)
                GT = F.vflip(GT)

            Transform = T.ColorJitter(brightness=0.2,contrast=0.2,hue=0.02)

            image = Transform(image)

            Transform =[]


        Transform.append(T.Resize((224,224)))
        Transform.append(T.ToTensor())
        Transform = T.Compose(Transform)
        
        image = Transform(image)
        GT = Transform(GT)

        Norm_ = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        image = Norm_(image)

        return image, GT

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.image_paths)

def get_loader(image_path, image_size, batch_size, num_workers=2, mode='train',augmentation_prob=0.4):
    """Builds and returns Dataloader."""
    
    dataset = ImageFolder(root = image_path, image_size =image_size, mode=mode,augmentation_prob=augmentation_prob)
    data_loader = DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    return data_loader

def get_dist_loader(image_path, image_size, batch_size, num_workers=2, mode='train',augmentation_prob=0.4):
    """Builds and returns Dataloader."""
    
    dataset = ImageFolder(root = image_path, image_size =image_size, mode=mode,augmentation_prob=augmentation_prob)
    dataset_sampler = torch.utils.data.distributed.DistributedSampler(SubsetRandomSampler(dataset))
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=False, \
        sampler=dataset_sampler,drop_last=True)
    return data_loader
