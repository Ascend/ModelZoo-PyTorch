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

from __future__ import print_function

import argparse
import os

import cv2
import numpy as np
import torchvision.transforms as transforms
from tqdm import tqdm


def process(args):
    testset_folder = args.dataset_folder
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    trans = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((104, 117, 123), (1, 1, 1)),
        ]
    )

    for dir_name, __, file_names in os.walk(testset_folder):
        for img_name in tqdm(file_names):
            sub_dir = dir_name.split("/")[-1]
            image_path = os.path.join(testset_folder, sub_dir, img_name)
            img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)

            target_size = 1000
            im_shape = img.shape
            im_size_max = np.max(im_shape[0:2])
            resize = target_size / im_size_max
            info = np.array(resize, dtype=np.float32)

            save_info_path = os.path.join(args.preinfo_folder, sub_dir)
            if not os.path.exists(save_info_path):
                os.makedirs(save_info_path)
            info.tofile(os.path.join(save_info_path, img_name.replace(".jpg", ".bin")))

            img = cv2.resize(
                img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_NEAREST
            )
            width_pad = target_size - img.shape[1]
            left = 0
            right = width_pad
            height_pad = target_size - img.shape[0]
            top = 0
            bottom = height_pad
            img = cv2.copyMakeBorder(
                img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )

            img = trans(img).unsqueeze(0)
            np.save(os.path.join(args.save_folder, img_name.replace(".jpg", "")), img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preprocess")
    parser.add_argument("--dataset-folder", default="data/widerface/val/images/")
    parser.add_argument("--preinfo-folder", default="widerface/prep_info")
    parser.add_argument("--save-folder", default="widerface/prep")
    args = parser.parse_args()
    process(args)
