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

import sys
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, datasets, models


def acc_eval(data_path = "./data/", pred_res_path = "./result/dumpOutput_device"+str(sys.argv[1]), output_path = "./result/acc_eval_res"):
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
    for images, labels in test_loader:                   
        for i in range(len(images)):
            cnt += 1
            image = images[i]
            label = labels[i]
            
            file_path = f"{pred_res_path}/image_{cnt}_1.txt"
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
            if(cnt % 100 == 99):
                print(f"current: {cnt}")
                
    accu = correct_cnt / (cnt + 1)
    print(f"Top-1 accuracy of inference: {accu * 100}%")
    print(f"Gap: {accu *100 / 0.954}%")
    with open(output_path, "w") as f:
        f.write(f"Top-1 accuracy of inference: {accu * 100}%\n")
        f.write(f"Gap: {accu *100 / 0.954}%\n")
        f.close()
            
if __name__ == "__main__":
    acc_eval()

