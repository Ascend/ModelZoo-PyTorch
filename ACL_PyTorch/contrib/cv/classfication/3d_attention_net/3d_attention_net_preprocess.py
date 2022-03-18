# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, datasets, models

def preprocess(data_path = "./data/", save_path = "./pre_process_result/"):
    # Image Preprocessing
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop((32, 32), padding=4),   #left, top, right, bottom
        transforms.ToTensor()
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    # when image is rgb, totensor do the division 255
    # CIFAR-10 Dataset
    train_dataset = datasets.CIFAR10(root=data_path,
                                   train=True,
                                   transform=transform,
                                   download=True)
    
    test_dataset = datasets.CIFAR10(root=data_path,
                                  train=False,
                                  transform=test_transform)
    
    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=64, # 64
                                               shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=32,
                                              shuffle=False)
    
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cnt = -1
    for images, labels in test_loader:
        for i in range(len(images)):
            cnt += 1
            image = images[i]
            label = labels[i]
            image = np.array(image).astype(np.float32)
            image.tofile(f"{save_path}image_{cnt}.bin")
            if(cnt % 100 == 9):
                print(f"current: {cnt}")
               
if __name__ == "__main__":
    preprocess()