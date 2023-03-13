# Copyright 2023 Huawei Technologies Co., Ltd
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
import torchvision.transforms as transforms
from PIL import Image


def preprocess(src_path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    preprocess_pipline = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    i = 0
    in_files = os.listdir(src_path)

    for files_name in in_files:
        files = os.listdir(os.path.join(src_path, files_name))
        for file in files:
            i = i + 1
            print(file, "===", i)

            input_image = Image.open(os.path.join(src_path, files_name, file)).convert('RGB')
            input_tensor = preprocess_pipline(input_image)
            img = np.array(input_tensor).astype(np.float32)
            img.tofile(os.path.join(save_path, file.split('.')[0] + ".bin"))


if __name__ == '__main__':
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    preprocess(input_path, output_path)
