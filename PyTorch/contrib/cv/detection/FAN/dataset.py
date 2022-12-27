# Copyright 2022 Huawei Technologies Co., Ltd
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

import random
from scipy import misc
from torch.utils.data import Dataset
import os
from skimage import io
import cv2
import numpy as np
from random import randint
from PIL import Image
import torch

class ImgDataset(Dataset):

    def __init__(self, dataset):
        self.x = []
        self.y = []
        imglist = os.listdir(dataset)
        for i, img in enumerate(imglist):
            if 'pts' not in img.split('.'):
                self.x.append(os.path.join(dataset, img))
                self.y.append(img)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        X = self.x[index]
        img = io.imread(X)
        Y = self.y[index]
        return img, Y
