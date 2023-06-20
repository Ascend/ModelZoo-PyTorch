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
import argparse
import multiprocessing
from PIL import Image
from tqdm import tqdm
import numpy as np


model_config = {
    'resize': 256,
    'centercrop': 224,
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
}


def center_crop(img, output_size):
    if isinstance(output_size, int):
        output_size = (int(output_size), int(output_size))
    image_width, image_height = img.size
    crop_height, crop_width = output_size
    crop_top = int(round((image_height - crop_height) / 2.))
    crop_left = int(round((image_width - crop_width) / 2.))
    return img.crop((crop_left, crop_top, crop_left + crop_width, crop_top + crop_height))


def resize(img, size, interpolation=Image.BILINEAR):
    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)


def gen_input_bin(file_batch):
    for file_name in tqdm(file_batch):
        # RGBA to RGB
        image = Image.open(file_name[1]).convert('RGB')
        image = resize(image, model_config['resize']) # Resize
        image = center_crop(image, model_config['centercrop']) # CenterCrop
        img = np.array(image, dtype=np.int8)
        img.tofile(os.path.join(save_path, file_name[0].split('.')[0] + ".bin"))


def preprocess(data_path):
    in_files = sorted(os.listdir(data_path))
    image_files = []
    for file_name in in_files:
        file_path = os.path.join(data_path, file_name)
        if os.path.isdir(file_path):
            image_files += [(image_name, 
                            os.path.join(file_path, image_name)) 
                            for image_name in os.listdir(file_path)]
        else:
            image_files.append((file_name, file_path))
    file_batches = [image_files[i:i + 500] for i in range(0, 50000, 500) if image_files[i:i + 500] != []]
    thread_pool = multiprocessing.Pool(len(file_batches))
    for batch in file_batches:
        thread_pool.apply_async(gen_input_bin, args=(batch, ))
    thread_pool.close()
    thread_pool.join()
    print("in thread, except will not report! please ensure bin files generated.")


def amct_input_bin():
    in_files = sorted(os.listdir(src_path))
    image_name = in_files[0]
    file_path = os.path.join(src_path, image_name)
    if os.path.isdir(file_path):
        image_name = os.listdir(file_path)[0]
        file_path = os.path.join(file_path, image_name)
    # RGBA to RGB
    image = Image.open(file_path).convert('RGB')
    image = resize(image, model_config['resize']) # Resize
    image = center_crop(image, model_config['centercrop']) # CenterCrop
    img = np.array(image, dtype=np.float32)
    img = img.transpose(2, 0, 1) # ToTensor: HWC -> CHW
    img = img / 255. # ToTensor: div 255
    img -= np.array(model_config['mean'], dtype=np.float32)[:, None, None] # Normalize: mean
    img /= np.array(model_config['std'], dtype=np.float32)[:, None, None] # Normalize: std
    img.tofile(os.path.join(save_path, image_name.split('.')[0] + ".bin"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='./ImageNet/val')
    parser.add_argument('--save_path', type=str, default='./prep_dataset')
    parser.add_argument('--amct', action='store_true')
    args = parser.parse_args()
    src_path = os.path.realpath(args.src_path)
    save_path = os.path.realpath(args.save_path)
    if not os.path.isdir(save_path):
        os.makedirs(os.path.realpath(save_path))
    if args.amct:
        amct_input_bin()
    else:
        preprocess(src_path)
