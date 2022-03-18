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

import os, sys
sys.path.append(r"./pytorch-ssd")
from vision.ssd.data_preprocessing import PredictionTransform
import cv2
import numpy as np


def pre_process(src_path, save_path):
    image_size = 300
    image_mean = np.array([127, 127, 127])  # RGB layout
    image_std = 128.0
    transform = PredictionTransform(image_size, image_mean, image_std)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    i = 0
    in_files = os.listdir(src_path)
    for file in in_files:
        i = i + 1
        print(file, "===", i)
        input_image = cv2.imread(src_path + '/' + file)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_tensor = transform(input_image)
        img = np.array(input_tensor).astype(np.float32)
        img.tofile(os.path.join(save_path, file.split('.')[0] + ".bin"))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python pre_process.py <src path>  <save path>')
        sys.exit(0)

    src_path = sys.argv[1]
    save_path = sys.argv[2]
    pre_process(src_path, save_path)