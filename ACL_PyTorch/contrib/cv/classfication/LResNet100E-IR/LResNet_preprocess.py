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
# limitations under the License

import os
import sys
from tqdm import trange
import argparse
sys.path.append("./LResNet")
import pickle
from glob import glob

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms as trans


def jpg2bin(data_path, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    transform = trans.Compose([
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    bins, issame_list = pickle.load(open(data_path, 'rb'), encoding='bytes')
    for i in trange(len(bins)):
        _bin = bins[i]
        img_np_arr = np.frombuffer(_bin, np.uint8)
        img = cv2.imdecode(img_np_arr, cv2.IMREAD_COLOR)
        img = Image.fromarray(img.astype(np.uint8))
        img = transform(img)
        img_flip = torch.flip(img, dims=[2])
        img = img.numpy()
        img_flip = img_flip.numpy()

        img.tofile(os.path.join(save_dir, str(i) + ".bin"))
        img_flip.tofile(os.path.join(save_dir, str(i) + "_flip" + ".bin"))


    np.save('data/lfw_list', np.array(issame_list))
    print('success load data')


def bin2info(bin_dir, data_info, width, height):
    bin_images = glob(os.path.join(bin_dir, '*.bin'))
    with open(data_info, 'w') as file:
        for index, img in enumerate(bin_images):
            content = ' '.join([str(index), img, width, height])
            file.write(content)
            file.write('\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--file_type', type=str, default="jpg")
    parser.add_argument('--data_path', type=str, default="./data/lfw.bin")
    parser.add_argument('--info_path', type=str, default="./data/lfw")
    parser.add_argument('--width;', type=str, default="112")
    parser.add_argument('--height;', type=str, default="112")


    args = parser.parse_args()

    file_type = args.file_type
    data_path = args.data_path
    info_path = args.info_path

    if file_type == 'bin':
        width = args.width
        height = args.height
    elif file_type == 'jpg':
        jpg2bin(data_path, info_path)

