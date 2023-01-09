# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import numpy as np
import cv2


def scale(img, long_size=2240):
    h, w = img.shape[0:2]
    scale = long_size * 1.0 / max(h, w)
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    # img = cv2.resize(img, (1260, 2240))
    print(img.shape)
    return img


def psenet_onnx(file_path, bin_path):
    if not os.path.exists(bin_path):
        os.makedirs(bin_path)
    i = 0
    in_files = os.listdir(file_path)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for file in in_files:
        i = i + 1
        print(file, "====", i)
        img = cv2.imread(os.path.join(file_path, file))
        img = img[:, :, [2, 1, 0]] # bgr -> rgb
        # img = scale(img)
        img = cv2.resize(img, (1216, 704))

        img = np.array(img, dtype=np.float32)
        img = img / 255.

        # 均值方差
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
    psenet_onnx(file_path, bin_path)
