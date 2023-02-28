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
import argparse
import multiprocessing

import numpy as np
from PIL import Image
from tqdm import tqdm


model_config = {
    224: {
        'resize': 248,
        'centercrop': 224,
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5],
    },
    384: {
        'resize': 384,
        'centercrop': 384,
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5],
    },
}


def parse_arguments():
    parser = argparse.ArgumentParser(description='Path', add_help=False)
    parser.add_argument('--image_size', required=True, type=int,
                        help='The output image size.')
    parser.add_argument('--data_path', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--store_path', metavar='DIR',
                        help='path to store')
    return parser.parse_args()


def center_crop(img, output_size):
    if isinstance(output_size, int):
        output_size = (int(output_size), int(output_size))
    image_width, image_height = img.size
    crop_height, crop_width = output_size
    crop_top = int(round((image_height - crop_height) / 2.))
    crop_left = int(round((image_width - crop_width) / 2.))
    return img.crop((crop_left, crop_top, crop_left + crop_width, crop_top + crop_height))


def resize(img, size, interpolation=Image.BICUBIC):
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


def normalize(img, means, variances):
    img = np.array(img)
    for channel, (mean, variance) in enumerate(zip(means, variances)):
        img[:, :, channel] = (img[:, :, channel] - mean) / variance
    return img


def get_input_bin(file_batches, batch, save_path, image_size):
    for file_name, file_path in tqdm(file_batches[batch]):
        image = Image.open(file_path).convert('RGB')
        image = resize(image, model_config[image_size]['resize']) # Resize
        image = center_crop(image, model_config[image_size]['centercrop']) # CenterCrop

        img = np.array(image).astype(np.float32) / 255
        img = normalize(img, model_config[image_size]['mean'], model_config[image_size]['std']) # Normalization
        img = img.transpose((2, 0, 1))
        img.tofile(os.path.join(save_path, file_name.split('.')[0] + ".bin"))


def preprocess(src_path, save_path, image_size):
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
        thread_pool.apply_async(get_input_bin, args=(file_batches, batch, save_path, image_size))

    thread_pool.close()
    thread_pool.join()
    print("In multi-threading, errors will not be reported! Please ensure bin files are generated correctly.")
    print("Done.")


if __name__ == "__main__":
    args = parse_arguments()

    if not os.path.exists(args.store_path):
        os.makedirs(args.store_path)

    preprocess(args.data_path, args.store_path, args.image_size)
