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

import os
import sys
import numpy as np
from PIL import Image


def resize(img, size, interpolation=Image.BILINEAR):
    if img.height <= img.width:
        ratio = size / img.height
        w_size = int(img.width * ratio)
        img = img.resize((w_size, size), interpolation)
    else:
        ratio = size / img.width
        h_size = int(img.height * ratio)
        img = img.resize((size, h_size), interpolation)

    return img


def center_crop(img, out_height, out_width):
    height, width, _ = img.shape
    left = int((width - out_width) / 2)
    right = int((width + out_width) / 2)
    top = int((height - out_height) / 2)
    bottom = int((height + out_height) / 2)
    img = img[top:bottom, left:right]
    return img


def deepmar_onnx(file_path, bin_path):
    in_files = os.listdir(file_path)
    if not os.path.exists(bin_path):
        os.makedirs(bin_path)
    i = 0
    input_size = 256
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for file in in_files:
        i = i + 1
        print(file, "====", i)
        img = Image.open(os.path.join(file_path, file)).convert('RGB')
        img = resize(img, input_size)  # transforms.Resize(256)
        img = np.array(img, dtype=np.float32)
        img = center_crop(img, 224, 224)   # transforms.CenterCrop(224)

        img = img / 255.  # transforms.ToTensor()

        # mean and variance
        img[..., 0] -= mean[0]
        img[..., 1] -= mean[1]
        img[..., 2] -= mean[2]
        img[..., 0] /= std[0]
        img[..., 1] /= std[1]
        img[..., 2] /= std[2]

        img = img.transpose(2, 0, 1) # HWC -> CHW
        img.tofile(os.path.join(bin_path, file.split('.')[0] + '.bin'))


if __name__ == "__main__":
    file_path = os.path.abspath(sys.argv[1])
    bin_path = os.path.abspath(sys.argv[2])
    deepmar_onnx(file_path, bin_path)
