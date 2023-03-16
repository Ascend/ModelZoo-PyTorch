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


from __future__ import absolute_import
from __future__ import division

import sys
import os
import argparse
from torchvision import datasets, transforms
from PIL import Image


def preprocess(root_path, bin_path):
    if not os.path.exists(bin_path):
        os.mkdir(bin_path)
    transform = transforms.Compose([
        transforms.Resize(256, Image.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(root_path, transform=transform)
    for index, (img_path, classes_number) in enumerate(dataset.imgs):
        img_data = dataset[index][0]
        file_name = f"{os.path.basename(img_path).split('.')[0]}.bin"
        des_path = os.path.join(bin_path, file_name)
        print(f'{index}/"{len(dataset)} : {des_path}"')
        img_data.numpy().tofile(des_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', default="data/imagenet/val")
    parser.add_argument('--bin_path', default='bin_path')
    args = parser.parse_args()
    preprocess(args.root_path, args.bin_path)