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
import argparse
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm
def preprocess(src_path, save_path):

    preprocess = transforms.Compose([
        transforms.Resize(342),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    i = 0
    in_files = os.listdir(src_path)
    for file in tqdm(in_files):
        i = i + 1
        input_image = Image.open(src_path + '/' + file).convert('RGB')
        input_tensor = preprocess(input_image)
        img = np.array(input_tensor).astype(np.float32)
        img.tofile(os.path.join(save_path, file.split('.')[0] + ".bin"))
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocess of MaskRCNN PyTorch model')
    parser.add_argument("--src_path", default="./coco2017/", help='image of dataset')
    parser.add_argument("--save_path", default="./coco2017_bin/", help='Preprocessed image buffer')
    flags = parser.parse_args()    

    src_path = os.path.realpath(flags.src_path)
    save_path = os.path.realpath(flags.save_path)
    if not os.path.isdir(save_path):
        os.makedirs(os.path.realpath(save_path))
    preprocess(src_path,save_path)