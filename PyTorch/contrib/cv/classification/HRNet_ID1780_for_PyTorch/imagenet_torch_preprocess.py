# Copyright 2020 Huawei Technologies Co., Ltd
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
import multiprocessing
from PIL import Image
import numpy as np


model_config = {
    'hrnet': {
        'resize': 256,
        'centercrop': 224,
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
    }
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


def gen_input_bin(mode_type_inp, file_batches, batch):
    i = 0
    for file in file_batches[batch]:
        i = i + 1
        # RGBA to RGB
        s = os.path.join(src_path,file)
        image = Image.open(os.path.join(src_path, file))#
        image = image.convert('RGB')
        image = resize(image, model_config[mode_type_inp]['resize']) # Resize
        image = center_crop(image, model_config[mode_type_inp]['centercrop']) # CenterCrop
        img = np.array(image, dtype=np.float32)
        img = img.transpose(2, 0, 1) # ToTensor: HWC -> CHW
        img = img / 255. # ToTensor: div 255
        img -= np.array(model_config[mode_type_inp]['mean'], dtype=np.float32)[:, None, None] # Normalize: mean
        img /= np.array(model_config[mode_type_inp]['std'], dtype=np.float32)[:, None, None] # Normalize: std
        img.tofile(os.path.join(save_path, file.split('.')[0] + ".bin"))


def preprocess(mode_type_inp, src_path_inp, save_path_inp):
    files = os.listdir(src_path_inp)
    file_batches = [files[i:i + 500] for i in range(0, 50000, 500) if files[i:i + 500] != []]
    thread_pool = multiprocessing.Pool(len(file_batches))
    for batch in range(len(file_batches)):
        thread_pool.apply_async(gen_input_bin, args=(mode_type_inp, file_batches, batch))
    thread_pool.close()
    thread_pool.join()
    print("in thread, except will not report! please ensure bin files generated.")


if __name__ == '__main__':
    if len(sys.argv) < 4:
        raise Exception("usage: python3 xxx.py [model_type] [src_path] [save_path]")
    mode_type = sys.argv[1]
    src_path = sys.argv[2]
    save_path = sys.argv[3]
    src_path = os.path.realpath(src_path)
    save_path = os.path.realpath(save_path)
    if mode_type not in model_config:
        model_type_help = "model type: "
        for key in model_config.keys():
            model_type_help += key
            model_type_help += ' '
        raise Exception(model_type_help)
    if not os.path.isdir(save_path):
        os.makedirs(os.path.realpath(save_path))
    preprocess(mode_type, src_path, save_path)
