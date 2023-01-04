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
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Path', add_help=False)
parser.add_argument('--data-path', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--store-path', metavar='DIR',
                    help='path to store')
parser.add_argument('--model-name', default='vit_base_patch32_224',
                    type=str, help='mode name for vit')
parser.add_argument('--dtype', default='float32',
                    type=str, help='dtype for input data')


def preprocess(src_path, save_path):
    model = timm.create_model(args.model_name)
    model.eval()
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    in_files = os.listdir(src_path)
    def process(image_path):
        file_name = os.path.basename(image_path)
        input_image = Image.open(image_path).convert('RGB')
        input_tensor = transform(input_image)
        img = np.array(input_tensor).astype(args.dtype)
        img.tofile(os.path.join(save_path, file_name.split('.')[0] + ".bin"))

    image_path_list = []
    for file_name in tqdm(in_files):
        file_path = os.path.join(src_path, file_name)
        if os.path.isdir(file_path):
            for image_name in os.listdir(file_path):
                image_path = os.path.join(file_path, image_name)
                image_path_list.append(image_path)
        else:
            image_path_list.append(file_path)
    for image_path in tqdm(image_path_list):
        process(image_path)


if __name__ == "__main__":
    args = parser.parse_args()
    preprocess(args.data_path, args.store_path)
