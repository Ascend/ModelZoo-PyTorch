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


import os
import numpy as np
import argparse
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.transforms import _str_to_pil_interpolation
from multiprocessing import Pool
from PIL import Image
from tqdm import tqdm


def build_transform(img_size, interpolation=InterpolationMode.BICUBIC):
    t = []
    t.append(
        transforms.Resize((img_size, img_size),
                          interpolation=interpolation))
    return transforms.Compose(t)


def make_transform(image_info):
    image_name, image_path = image_info
    input_image = Image.open(image_path).convert('RGB')
    input_tensor = transforms(input_image)
    img = np.array(input_tensor).astype(args.dtype)
    img.tofile(os.path.join(args.out_dir, image_name.split('.')[0] + ".bin"))


def preprocess(args):
    dataset_dir = args.input_dir
    save_dir = args.out_dir
    num_worker = args.num_worker

    val_files = os.listdir(dataset_dir)
    image_infos = []
    for file_name in val_files:
        file_path = os.path.join(dataset_dir, file_name)
        if os.path.isdir(file_path):
            image_infos += [(image_name, os.path.join(file_path, image_name)) \
                           for image_name in os.listdir(file_path)]
        else:
            image_infos.append((file_name, file_path))

    with Pool(num_worker) as p:
        res = list(tqdm(p.imap(make_transform, image_infos), total=len(image_infos)))


def parse_arguments():
    parser = argparse.ArgumentParser(description='SwinTransformer preprocess.')
    parser.add_argument('-s', '--img_size', type=int, default=384,
                        help='input image size for swintransformer model')
    parser.add_argument('-i', '--input_dir', type=str, required=True,
                        help='input dataset dir')
    parser.add_argument('-o', '--out_dir', type=str, required=True,
                        help='output dir for preprocessed data')
    parser.add_argument('-n', '--num_worker', type=int, default=16,
                        help='num of workers for preprocess data')
    parser.add_argument('-d', '--dtype', type=str, default="uint8",
                        help='dtype for preprocessed data')
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    return args


if __name__ == '__main__':
    args = parse_arguments()
    transforms = build_transform(args.img_size)
    preprocess(args)
