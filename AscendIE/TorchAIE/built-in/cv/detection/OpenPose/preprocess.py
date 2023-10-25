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
import math
import argparse
import torch
import numpy as np
import cv2
from tqdm import tqdm

sys.path.append("./lightweight-human-pose-estimation.pytorch")
from val import normalize


def pad_width(img, stride, pad_value, min_dims, name, height, width, pad_txt_path):
    h, w, _ = img.shape
    min_dims[0] = math.ceil(min_dims[0] / float(stride)) * stride
    min_dims[1] = math.ceil(min_dims[1] / float(stride)) * stride
    pad = [
        int(math.floor((min_dims[0] - h) / 2.0)),
        int(math.floor((min_dims[1] - w) / 2.0)),
    ]
    pad.append(int(min_dims[0] - h - pad[0]))
    pad.append(int(min_dims[1] - w - pad[1]))
    padded_img = cv2.copyMakeBorder(
        img,
        pad[int(0)],
        pad[int(2)],
        pad[int(1)],
        pad[int(3)],
        cv2.BORDER_CONSTANT,
        value=pad_value,
    )
    with open(pad_txt_path, "a") as f:
        f.write(
            str(name)
            + " "
            + str(height)
            + " "
            + str(width)
            + " "
            + str(pad[int(0)])
            + " "
            + str(pad[int(2)])
            + " "
            + str(pad[int(1)])
            + " "
            + str(pad[int(3)])
            + "\n"
        )
    return padded_img


def image_preprocess(
    img,
    name,
    pad_txt_path,
    base_height=368,
    base_width=640,
    stride=8,
    cpu=True,
    pad_value=(0, 0, 0),
    img_mean=np.array([128, 128, 128], np.float32),
    img_scale=np.float32(1 / 256),
):
    norm_img = normalize(img, img_mean, img_scale)
    height, width, _ = img.shape
    height_scale = base_height / height
    width_scale = base_width / width
    scale = min(height_scale, width_scale)
    scaled_img = cv2.resize(
        norm_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR
    )
    min_dims = [base_height, base_width]
    padded_img = pad_width(
        scaled_img, stride, pad_value, min_dims, name, height, width, pad_txt_path
    )
    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cuda()
    return tensor_img


def preprocess(raw_img_path, processed_img_path, pad_txt_path):
    in_files = os.listdir(raw_img_path)
    for file in tqdm(in_files):
        img_path = os.path.join(raw_img_path, file)
        input_image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        input_tensor = image_preprocess(input_image, file, pad_txt_path)
        img = np.array(input_tensor).astype(np.float32)
        img.tofile(os.path.join(processed_img_path, file.split(".")[0] + ".bin"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-img-path",
        type=str,
        default="/data/datasets/coco/val2017",
        help="the source path of images",
    )
    parser.add_argument(
        "--processed-img-path",
        type=str,
        default="datasets/coco/processed_img",
        help="the path of saving bin of each image",
    )
    parser.add_argument(
        "--pad-txt-path",
        type=str,
        default="./output/pad.txt",
        help="the path of pad.txt saving the info of padding",
    )
    args = parser.parse_args()
    with open(args.pad_txt_path, "a+") as f:
        f.truncate(0)
    preprocess(args.raw_img_path, args.processed_img_path, args.pad_txt_path)


if __name__ == "__main__":
    main()
