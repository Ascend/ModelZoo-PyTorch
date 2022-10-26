# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================

import torch
import os
import math
import argparse
import numpy as np
import PIL
from tqdm import tqdm

from PIL import Image
from torchvision import transforms


def preprocess(img):
    input_transform = transforms.Compose([
        transforms.Resize(size=256, interpolation=PIL.Image.BICUBIC),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return input_transform(img)


def img_preprocess(args):
    save_path = os.path.realpath(args.prep_image)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    in_files = os.listdir(args.image_path)
    file_list = []
    if not os.path.isfile(os.path.join(args.image_path, in_files[0])):
        for sub_dir in in_files:
            image_path = os.path.join(args.image_path, sub_dir)
            sub_file_list = os.listdir(image_path)
            for file in sub_file_list:
                file_list.append(os.path.join(image_path, file))
    else:
        for file in in_files:
            file_list.append(os.path.join(args.image_path, file))

    suffix_len = -5
    file_list.sort(key=lambda x: int(x[suffix_len-8:suffix_len]))
    for i in tqdm(range(int(np.ceil(len(file_list) / args.batch_size)))):

        for idx in range(args.batch_size):
            file_index = i * args.batch_size + idx
            if file_index < len(file_list):
                file = file_list[file_index]
                input_image = Image.open(file).convert('RGB')
                image_tensor = preprocess(input_image).unsqueeze(0)
            else:
                image_tensor = torch.zeros([1, 3, 224, 224])

            input_tensor = image_tensor if idx == 0 \
                else torch.cat([input_tensor, image_tensor], dim=0)

        img = np.array(input_tensor).astype(np.float32)
        img.tofile(os.path.join(save_path, "input_{:05d}.bin".format(i)))


#============================================================================
# Main
#============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default="/opt/npu/imageNet/val")
    parser.add_argument('--prep_image', type=str, default="./prep_image_bs8")
    parser.add_argument('--batch_size', type=int, default=8)
    opt = parser.parse_args()

    img_preprocess(opt)


