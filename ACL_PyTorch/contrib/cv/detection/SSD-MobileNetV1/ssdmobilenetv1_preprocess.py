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

import tqdm
import cv2
import numpy as np

sys.path.append(r"./pytorch-ssd")
from vision.ssd.data_preprocessing import PredictionTransform


def preprocess(src_path, save_path):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    image_size = 300
    image_mean = np.array([127, 127, 127])  # RGB layout
    image_std = 128.0
    transform = PredictionTransform(image_size, image_mean, image_std)
    
    in_files = os.listdir(src_path)
    for file in tqdm.tqdm(in_files):
        input_image = cv2.imread(os.path.join(src_path, file))
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_tensor = transform(input_image)
        img = np.array(input_tensor).astype(np.float32)
        img.tofile(os.path.join(save_path, file.split('.', 1)[0] + ".bin"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser('data preprocess.')
    parser.add_argument('--src_path', type=str, required=True, 
                        help='path to original dataset.')
    parser.add_argument('--save_path', type=str, required=True, 
                        help='a directory to save bin files.')
    args = parser.parse_args()

    preprocess(args.src_path, args.save_path)
