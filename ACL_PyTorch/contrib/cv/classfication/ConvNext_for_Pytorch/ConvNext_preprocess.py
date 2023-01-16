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
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import os
import numpy as np
import torch
import argparse
from tqdm import tqdm

def preprocess(save_path, src_path):
    preprocess = transforms.Compose([
        transforms.Resize(256, InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    for root, dirs, files in os.walk(src_path):
        for file in files:
            pbar.update(1)
            input_image = Image.open(os.path.join(root, file)).convert('RGB')
            input_tensor = preprocess(input_image)
            img = np.array(input_tensor).astype(np.float32)
            img.tofile(os.path.join(save_path, file.split('.')[0] + ".bin"))


if __name__ == '__main__':       
    parser = argparse.ArgumentParser('ConvNeXt training and evaluation script for image classification', add_help=False)
    parser.add_argument('--dataset_root', default='', type=str)
    parser.add_argument('--output_dir', default='', type=str)
    parser.add_argument('--bs', default='1', type=str)

    args = parser.parse_args()
    pbar = tqdm(total=50000) 
    preprocess(args.output_dir, args.dataset_root)
    pbar.close()