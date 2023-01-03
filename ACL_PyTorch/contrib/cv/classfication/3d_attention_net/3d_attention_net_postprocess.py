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

import argparse
import sys
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, datasets, models


def acc_eval(pred_res_path, data_path, output_path, prefix):
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    test_dataset = datasets.CIFAR10(root=data_path,
                                  train=False,
                                  transform=test_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=32,
                                              shuffle=False)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    cnt = -1
    correct_cnt = 0
    for images, labels in tqdm(test_loader):
        for i in range(len(images)):
            cnt += 1
            image = images[i]
            label = labels[i]

            file_path = f"{pred_res_path}/{prefix}{cnt}_0.txt"
            with open(file_path, "r") as f:
                data = f.readline()
                temp = data.strip().split(" ")
                n_label = len(temp)
                data_vec = np.zeros((n_label), dtype=np.float32)
                for ind, prob in enumerate(temp):
                    data_vec[ind] = np.float32(prob)
                pred_label = np.argmax(data_vec)

            if(label == pred_label):
                correct_cnt += 1

    accu = correct_cnt / (cnt + 1)
    print(f"Top-1 accuracy of inference: {accu * 100}%")
    print(f"Gap: {accu *100 / 0.954}%")
    with open(output_path, "w") as f:
        f.write(f"Top-1 accuracy of inference: {accu * 100}%\n")
        f.write(f"Gap: {accu *100 / 0.954}%\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='3D Attention Net postprocess script')
    parser.add_argument('--pred_res_path', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--prefix', default="image_", type=str)
    opt = parser.parse_args()
    acc_eval(opt.pred_res_path, opt.data_path, opt.output_path, opt.prefix)
