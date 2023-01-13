# Copyright (C) Huawei Technologies Co., Ltd 2022-2022. All rights reserved.
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
import argparse

from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import numpy as np


def preprocess(src_path, save_path):

    in_files = sorted(os.listdir(src_path))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    preprocesser = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    for file in tqdm(in_files):
        input_image = Image.open(os.path.join(src_path, file)).convert('RGB')
        input_tensor = preprocesser(input_image)
        img = np.array(input_tensor).astype(np.float32)
        img.tofile(os.path.join(save_path, file.split('.')[0] + ".bin"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--src-path', type=str, default = None)
    parser.add_argument('--save-path', type=str, default = None)
    args = parser.parse_args()
    preprocess(args.src_path, args.save_path)