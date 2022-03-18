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
import sys
from PIL import Image
import numpy as np
import multiprocessing
import torchvision
import torchvision.transforms as transforms

MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2023, 0.1994, 0.2010)
transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def preprocess(src_path, save_path):
    valset = torchvision.datasets.CIFAR10(root=src_path, train=False, download=False, transform=transform_val)  
    labelpath = os.path.join(save_path, 'val_label.txt')
    labelfile = open(labelpath, 'w')
    for i in range(len(valset)):
        index = str(i).zfill(5)
        (np.array(valset[i][0]).astype(np.float32)).tofile(os.path.join(save_path, index + ".bin"))
        labelfile.write(str(index)+'.bin')
        labelfile.write(' ')
        labelfile.write(str(valset[i][1]))
        labelfile.write('\n')
    labelfile.close()    


if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise Exception("usage: python3 xxx.py [src_path] [save_path]")
    src_path = sys.argv[1]
    save_path = sys.argv[2]
    src_path = os.path.realpath(src_path)
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.realpath(save_path)
    preprocess(src_path, save_path)