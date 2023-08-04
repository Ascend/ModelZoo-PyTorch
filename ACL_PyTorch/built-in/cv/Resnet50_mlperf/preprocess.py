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
import multiprocessing
import cv2
import numpy as np
from tqdm import tqdm


model_config = {
    'resize': 224,
    'centercrop': 224,
    'mean': [123.675, 116.28, 103.53]
}


def center_crop(img, out_height, out_width):
    height, width, _ = img.shape
    left = int((width - out_width) / 2)
    right = int((width + out_width) / 2)
    top = int((height - out_height) / 2)
    bottom = int((height + out_height) / 2)
    img = img[top:bottom, left:right]
    return img


def resize_with_aspectratio(img, size, scale=87.5, inter_pol=cv2.INTER_LINEAR):
    if isinstance(size, int):
        height, width, _ = img.shape
        new_height = int(100. * size / scale)
        new_width = int(100. * size / scale)
        if height > width:
            w = new_width
            h = int(new_height * height / width)
        else:
            h = new_height
            w = int(new_width * width / height)
        img = cv2.resize(img, (w, h), interpolation=inter_pol)
        return img
    else:
        img = img.resize(size[::-1], interpolation)
        return img


def gen_input_bin(file_batches, batch, src_path, save_path):
    for file_name in tqdm(file_batches[batch]):
        image = cv2.imread(os.path.join(src_path, file_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = resize_with_aspectratio(image, model_config['resize'], inter_pol=cv2.INTER_AREA)
        img = center_crop(img, model_config['centercrop'], model_config['centercrop'])
        img = np.asarray(img, dtype='uint8')
        img.tofile(os.path.join(save_path, file_name.split('.')[0] + ".bin"))


def preprocess(source_path, dest_path):
    files = os.listdir(source_path)
    files.sort()
    if len(files) < 500:
        file_batches = [files[0: len(files)]]
    else:
        file_batches = [files[i:i + 500] for i in range(0, len(files), 500) if files[i:i + 500] != []]
    thread_pool = multiprocessing.Pool(len(file_batches))
    for batch in range(len(file_batches)):
        thread_pool.apply_async(gen_input_bin, args=(file_batches, batch, source_path, dest_path))
    thread_pool.close()
    thread_pool.join()
    print("in thread, except will not report! please ensure bin files generated.")


def amct_input_bin(src_path, save_path):
    in_files = os.listdir(src_path)
    image_names = in_files[0: 64]
    data = []
    for image_name in tqdm(image_names):
        file_path = os.path.join(src_path, image_name)
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = resize_with_aspectratio(image, model_config['resize'], inter_pol=cv2.INTER_AREA)
        img = center_crop(img, model_config['centercrop'], model_config['centercrop'])
        img = np.asarray(img, dtype='float32')
        img -= np.array(model_config['mean'], dtype='float32')
        img = img.transpose([2, 0, 1])
        data.append(img)
    batch_data = np.stack(data, axis=0)
    batch_data.tofile(os.path.join(save_path, image_name.split('.')[0] + ".bin"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='./ImageNet/val', help='path to images.')
    parser.add_argument('--save_path', type=str, default='./rep_dataset', help='path to save bin files.')
    parser.add_argument('--amct', action='store_true', help='if True, will generate quantization data.')
    args = parser.parse_args()

    if not os.path.isdir(args.save_path):
        os.makedirs(os.path.realpath(args.save_path))
    if args.amct:
        amct_input_bin(args.src_path, args.save_path)
    else:
        preprocess(args.src_path, args.save_path)


if __name__ == '__main__':
    main()

    