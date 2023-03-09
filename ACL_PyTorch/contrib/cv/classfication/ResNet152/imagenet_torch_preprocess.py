"""
Copyright 2023 Huawei Technologies Co., Ltd

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
import argparse
import multiprocessing

import numpy as np
from PIL import Image
from tqdm import tqdm


model_config = {
    'resnet': {
        'resize': 256,
        'centercrop': 256,
    },
}


def parse_arguments():
    parser = argparse.ArgumentParser(description='Path', add_help=False)
    parser.add_argument('--model_name', default="resnet", type=str,
                        help='Name of model.')
    parser.add_argument('--data_path', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--save_dir', metavar='DIR',
                        help='path to store outputs')
    return parser.parse_args()


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


def get_input_bin(file_batches, batch, save_path, model_name):
    for file_name, file_path in tqdm(file_batches[batch]):
        image = Image.open(file_path).convert('RGB')
        image = resize(image, model_config[model_name]['resize']) # Resize
        image = center_crop(image, model_config[model_name]['centercrop']) # CenterCrop

        img = np.array(image).astype(np.int8)
        img.tofile(os.path.join(save_path, file_name.split('.')[0] + ".bin"))


def preprocess(src_path, save_path, model_name):
    files = os.listdir(src_path)
    image_infos = []
    for file_name in files:
        file_path = os.path.join(src_path, file_name)
        if os.path.isdir(file_path):
            image_infos += [(image_name, os.path.join(file_path, image_name)) \
            for image_name in os.listdir(file_path)]
        else:
            image_infos.append((file_name, file_path))

    image_infos.sort()
    if len(image_infos) < 500:
        file_batches = [image_infos[0 : len(image_infos)]]
    else:
        file_batches = [image_infos[i:i + 500] for i in range(0, len(image_infos), 500) if image_infos[i:i + 500] != []]

    thread_pool = multiprocessing.Pool(len(file_batches))
    for batch in range(len(file_batches)):
        thread_pool.apply_async(get_input_bin, args=(file_batches, batch, save_path, model_name))

    thread_pool.close()
    thread_pool.join()
    print("In multi-threading, errors will not be reported! Please ensure bin files are generated correctly.")
    print("Done.")


if __name__ == "__main__":
    args = parse_arguments()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    preprocess(args.data_path, args.save_dir, args.model_name)
