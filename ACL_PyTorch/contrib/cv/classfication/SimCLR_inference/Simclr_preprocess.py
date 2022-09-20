"""
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
"""

import numpy as np
from torchvision import transforms
import torch
from torch import nn
import os
from PIL import Image
import sys
import pickle

class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img


def preprocess(srcfile_path, savefile_path):
    """ data preprocess """
    size = 32
    s = 1
    n_views = 2
    file_num = 0
    data = []
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomApply([color_jitter], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                          GaussianBlur(kernel_size=int(0.1 * size)),
                                          transforms.ToTensor()])
    if not os.path.exists(savefile_path):
        os.mkdir(savefile_path)
    with open(srcfile_path, "rb") as f:
        entry = pickle.load(f, encoding='latin1')
        data.append(entry['data'])
        images = np.vstack(data).reshape(-1, 3, 32, 32)
        images = np.transpose(images, (0, 2, 3, 1))
        for i in range(images.shape[0]):
            image = [data_transforms(Image.fromarray(images[i])) for j in range(n_views)]
            file_path = os.path.join(savefile_path, "Simclr_prep_" + str(file_num) + ".bin")
            file_num = file_num + 1
            image_file = np.array(image[0]).astype(np.int8)
            image_file.tofile(file_path)
            file_path = os.path.join(savefile_path, "Simclr_prep_" + str(file_num) + ".bin")
            image_file = np.array(image[1]).astype(np.int8)
            image_file.tofile(file_path)
            file_num = file_num + 1


if __name__ == "__main__":
    src_path = sys.argv[1]
    save_path = sys.argv[2]
    preprocess(src_path, save_path)
