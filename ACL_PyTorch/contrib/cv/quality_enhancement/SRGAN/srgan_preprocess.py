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

import os
import sys
import argparse

import numpy as np
from PIL import Image
from torchvision.transforms import transforms, ToTensor

parser = argparse.ArgumentParser(description="SRCNN preprocess script")
parser.add_argument("--src_path", default="./datasets/Set5_X2/", type=str,
                    help="path of source image files")
parser.add_argument("--save_path", default="./preprocess_data", type=str,
                    help='path of output')
args_parser = parser.parse_args()


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def preprocess(args):
    src_path = args.src_path
    save_path = args.save_path

    if not os.path.exists(src_path):
        print(f'input image path: {src_path} is not exist')
        sys.exit()

    lr_filenames = os.listdir(src_path)
    
    for index, image_file in enumerate(lr_filenames):
        lr_image = Image.open(os.path.join(src_path, image_file)).convert('RGB')
        width, height = lr_image.size
              
        bin_path = os.path.join(save_path, f'img_{width}_{height}')
        if not os.path.isdir(bin_path):
            os.makedirs(bin_path)
        lr_image = transforms.ToTensor()(lr_image).unsqueeze(0)
        img = to_numpy(lr_image)
        
        img.tofile(os.path.join(bin_path, image_file.split(".")[-2] + ".bin"))


if __name__ == '__main__':
    preprocess(args_parser)
