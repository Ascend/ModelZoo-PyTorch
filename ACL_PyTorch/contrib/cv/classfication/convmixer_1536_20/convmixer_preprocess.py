# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import os
import math
import torch
import argparse
import numpy as np

from PIL import Image
from torchvision import transforms


#============================================================================
# Functions
#============================================================================
def preprocess(img, img_size, crop_pct):
    scale_size = int(math.floor(img_size / crop_pct))
    input_transform = transforms.Compose([
        transforms.Resize(scale_size, Image.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return input_transform(img)


def img_preprocess(args):
    save_path = os.path.realpath(args.prep_image)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    in_files = os.listdir(args.image_path)
    file_list = []
    if not os.path.isfile(in_files[0]):
        for sub_dir in in_files:
            image_path = os.path.join(args.image_path, sub_dir)
            sub_file_list = os.listdir(image_path)
            for file in sub_file_list:
                file_list.append(os.path.join(image_path, file))
    else:
        for file in in_files:
            file_list.append(os.path.join(args.image_path, file))

    suffix_len = -5
    file_list.sort(key=lambda x:int(x[suffix_len-8:suffix_len]))
    for i in range(int(np.ceil(len(file_list) / args.batch_size))):
        if i % 100 == 0:
            print("has generated input {:05d}...".format(i*args.batch_size))

        for idx in range(args.batch_size):
            file_index = i * args.batch_size + idx
            if file_index < len(file_list):
                file = file_list[file_index]
                input_image = Image.open(file).convert('RGB')
                image_tensor = preprocess(input_image, 224, 0.96).unsqueeze(0)
            else:
                image_tensor = torch.zeros([1,3,224,224])

            input_tensor = image_tensor if idx == 0 \
                else torch.cat([input_tensor, image_tensor], dim=0)

        img = np.array(input_tensor).astype(np.float32)
        img.tofile(os.path.join(save_path, "input_{:05d}.bin".format(i)))


#============================================================================
# Main
#============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path', type=str, default="/opt/npu/imageNet/val")
    parser.add_argument('--prep-image', type=str, default="./prep_image_bs1")
    parser.add_argument('--batch-size', type=int, default=1)
    opt = parser.parse_args()

    img_preprocess(opt)
