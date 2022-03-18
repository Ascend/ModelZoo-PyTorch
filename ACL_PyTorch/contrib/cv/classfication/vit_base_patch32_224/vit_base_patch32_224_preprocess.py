"""
Copyright 2020 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


import os
from PIL import Image
import numpy as np
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import argparse


parser = argparse.ArgumentParser(description='Path', add_help=False)
parser.add_argument('--data-path', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--store-path', metavar='DIR',
                    help='path to store')


def preprocess(src_path, save_path):
    os.mkdir(save_path)
    model = timm.create_model('vit_base_patch32_224')
    model.eval()
    config = resolve_data_config({},model=model)
    transform = create_transform(**config)
    i = 0
    in_files = os.listdir(src_path)
    for file in in_files:
        i = i + 1
        print(file, "===", i)
        input_image = Image.open(src_path+'/'+file).convert('RGB')
        input_tensor = transform(input_image)
        img = np.array(input_tensor).astype(np.float32)
        img.tofile(os.path.join(save_path, file.split('.')[0] + ".bin"))


if __name__ == "__main__":
    args = parser.parse_args()
    preprocess(args.data_path,args.store_path)