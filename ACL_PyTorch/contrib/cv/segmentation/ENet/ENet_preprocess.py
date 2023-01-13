# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import numpy as np

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

from PIL import Image
from torchvision import transforms


def _val_sync_transform(img, mask):
    outsize = 480
    short_size = outsize
    w, h = img.size
    if w > h:
        oh = short_size
        ow = int(1.0 * w * oh / h)
    else:
        ow = short_size
        oh = int(1.0 * h * ow / w)
    img = img.resize((ow, oh), Image.BILINEAR)
    mask = mask.resize((ow, oh), Image.NEAREST)
    # center crop
    w, h = img.size
    x1 = int(round((w - outsize) / 2.))
    y1 = int(round((h - outsize) / 2.))
    img = img.crop((x1, y1, x1 + outsize, y1 + outsize))
    mask = mask.crop((x1, y1, x1 + outsize, y1 + outsize))
    # final transform
    img, mask = np.array(img), np.array(mask).astype('int32')
    return img, mask

def _get_city_pairs(folder, split='val'):
    def get_path_pairs(img_folder, mask_folder):
        img_paths = []
        mask_paths = []
        for root, _, files in os.walk(img_folder):
            for filename in files:
                if filename.endswith('.png'):
                    imgpath = os.path.join(root, filename)
                    foldername = os.path.basename(os.path.dirname(imgpath))
                    maskname = filename.replace('leftImg8bit', 'gtFine_labelIds')
                    maskpath = os.path.join(mask_folder, foldername, maskname)
                    if os.path.isfile(imgpath) and os.path.isfile(maskpath):
                        img_paths.append(imgpath)
                        mask_paths.append(maskpath)
                    else:
                        print('cannot find the mask or image:', imgpath, maskpath)
        print('Found {} images in the folder {}'.format(len(img_paths), img_folder))
        return img_paths, mask_paths

    if split in ('train', 'val'):
        img_folder = os.path.join(folder, 'leftImg8bit/' + split)
        mask_folder = os.path.join(folder, 'gtFine/' + split)
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
        return img_paths, mask_paths

def preprocess(args):
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])

    images, mask_paths = _get_city_pairs(args.src_path, 'val')

    for i, image in enumerate(images):
        img = Image.open(image).convert('RGB')
        mask = Image.open(mask_paths[i])
        img, mask = _val_sync_transform(img, mask)
        img = input_transform(img)
        #img = np.asarray(img).astype(np.float16)

        img = np.asarray(img)

        filename = os.path.basename(image)

        img.tofile(os.path.join(args.save_path, os.path.splitext(filename)[0] + ".bin"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, default='prep_dataset')

    args = parser.parse_args()

    if not os.path.isdir(args.save_path):
        os.makedirs(os.path.realpath(args.save_path))
    preprocess(args)

# python ENet_preprocess.py --src_path=/root/.torch/datasets/citys --save_path prep_dataset