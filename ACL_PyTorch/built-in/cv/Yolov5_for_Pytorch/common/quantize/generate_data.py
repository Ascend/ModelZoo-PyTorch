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

import sys
import os
import cv2
import numpy as np
import argparse


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def preprocess(img_info_file, save_path, batch_size, img_size):
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
            output_data.tofile("{}/img_bs{}.bin".format(save_path, batch_size))


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
    parser.add_argument('--img_size', type=int, default=640, help='input data size')
    args = parser.parse_args()

    print(os.path.abspath(args.img_info_file))
    print(os.path.abspath(args.save_path))
    preprocess(args.img_info_file, args.save_path, args.batch_size, args.img_size)
