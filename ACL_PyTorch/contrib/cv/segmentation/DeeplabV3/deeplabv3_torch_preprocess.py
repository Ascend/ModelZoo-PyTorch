# Copyright 2021 Huawei Technologies Co., Ltd
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

import sys
import os
import cv2
import numpy as np
from torchvision import transforms


class Normalize(object):
    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, img):
        img = img.copy().astype(np.float32)
        # cv2 inplace normalization does not accept uint8
        assert img.dtype != np.uint8
        mean = np.float64(self.mean.reshape(1, -1))
        stdinv = 1 / np.float64(self.std.reshape(1, -1))
        if self.to_rgb:
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
        cv2.subtract(img, mean, img)  # inplace
        cv2.multiply(img, stdinv, img)  # inplace
        return img


def preprocess(src_path, save_path):
    """
    resnet50 pytorch preprocess
    """
    preprocess = transforms.Compose([
        Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
        transforms.ToTensor(),
    ])

    root = src_path

    # 扫描文件夹下所有文件
    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                rel_path = os.path.relpath(entry.path, root)
                if suffix is None or rel_path.endswith(suffix):
                    yield rel_path
            elif recursive and os.path.isdir(entry.path):
                # scan recursively if entry.path is a directory
                yield from _scandir(
                    entry.path, suffix=suffix, recursive=recursive)

    in_files = _scandir(src_path, '_leftImg8bit.png', True)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    i = 0
    for file in in_files:
        i = i + 1
        print(file, "====", i)
        input_image = cv2.imread(src_path + '/' + file)
        input_tensor = preprocess(input_image)
        # print(file.split('/')[-1].split('.')[0])
        # print(input_tensor)
        img = np.array(input_tensor).astype(np.float32)
        # print(img.shape)
        img.tofile(os.path.join(save_path, file.split('/')[-1].split('.')[0] + ".bin"))


if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise Exception("usage: python3 xxx.py [src_path] [save_path]")
    src_path = sys.argv[1]
    save_path = sys.argv[2]
    src_path = os.path.realpath(src_path)
    save_path = os.path.realpath(save_path)
    preprocess(src_path, save_path)
