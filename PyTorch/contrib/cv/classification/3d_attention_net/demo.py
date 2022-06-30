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

from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision
from torchvision import transforms, datasets, models
import os
if torch.__version__ >= "1.8.1":
    import torch_npu
else:
    import torch.npu
#import cv2
import time
from collections import OrderedDict
from model.residual_attention_network import ResidualAttentionModel_92_32input_update as ResidualAttentionModel



def main():
    model_file = 'model_92_sgd.pkl' # 执行完训练任务后，程序自动保存的模型文件

    # Image Preprocessing
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop((32, 32), padding=4),
        transforms.ToTensor()
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    test_dataset = datasets.CIFAR10(root='./data/', train=False, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=512, shuffle=False)

    model = ResidualAttentionModel(10)
    base_weights = torch.load(model_file, map_location="cpu")
    print('Loading base network...')
    new_state_dict = OrderedDict()
    for k, v in base_weights.items():
        if(k[0: 7] == "module."):
            name = k[7:]
        else:
            name = k[0:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    cnt = 0
    model.eval()
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        for i in range(len(labels.data)):
            cnt += 1
            print(f"Image{cnt}   real_class: {labels.data[i]}   pred_clss: {predicted[i]}")

if __name__ == "__main__":
    main()
                    