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
import argparse

def preprocess(img_info_file, save_path, batch_size):
    in_files = []
    output_data = []
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(img_info_file, 'r') as file:
        contents = file.read().split('\n')
    for i in contents[:-1]:
        in_files.append(i.split()[1])

    for i, file in enumerate(in_files):
        img0 = cv2.imread(file)
        resized = cv2.resize(img0, (640, 640), interpolation=cv2.INTER_LINEAR)
        input_data = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        input_data = np.transpose(input_data, (2, 0, 1)).astype(np.float32)
        input_data = np.expand_dims(input_data, axis=0)
        input_data /= 255.0
        print("shape:", input_data.shape)

        if i % batch_size == 0:
            output_data = input_data
        else:
            output_data = np.concatenate((output_data, input_data), axis=0)

        if (i + 1) % batch_size == 0:
            output_data.tofile("{}/img_bs{}_n{}.bin".format(save_path, batch_size, i))


if __name__ == "__main__":
    """
    python3 generate_data.py \
            --img_info_file=img_info_amct.txt \
            --save_path=amct_data \
            --batch_size=1
    """
    parser = argparse.ArgumentParser(description='YoloV5 offline model inference.')
    parser.add_argument('--img_info_file', type=str, default="img_info_amct.txt", help='original data')
    parser.add_argument('--save_path', type=str, default="./amct_data", help='data for amct')
    parser.add_argument('--batch_size', type=int, default=1, help='om batch size')
    args = parser.parse_args()

    print(os.path.abspath(args.img_info_file))
    print(os.path.abspath(args.save_path))
    preprocess(args.img_info_file, args.save_path, args.batch_size)
