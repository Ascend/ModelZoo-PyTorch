#!/usr/bin/env python
# coding=utf-8

"""
 Copyright(C) 2022. Huawei Technologies Co.,Ltd. All rights reserved.

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
import pickle as pickle
import argparse
import numpy as np
from PIL import Image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=0)
    parser.add_argument('--file_path', type=str, default='../data/input/peta/images')
    parser.add_argument('--result_path', type=str, default='./binfile')
    args = parser.parse_args()

    num = args.num
    file_path = args.file_path
    result_path = args.result_path
    # load data
    if not (os.path.exists(result_path)):
        os.mkdir(result_path)
    image = []
    label = []
    with open("../data/config/peta_dataset.pkl", 'rb') as data_file:
        dataset = pickle.load(data_file)
    with open("../data/config/peta_partition.pkl", 'rb') as data_file:
        partition = pickle.load(data_file)
    for idx in partition['test'][num]:
        image.append(dataset['image'][idx])
        label_tmp = np.array(dataset['att'][idx])[dataset['selected_attribute']].tolist()
        label.append(label_tmp)

    # Filter the test pictures in the dataset
    jpg_name = os.listdir(file_path)
    valid_img = []
    label_selected = []
    str_num = 5
    for name in image:
        if name[0:str_num] + '.jpg' not in jpg_name:
            continue
        im = Image.open('../data/input/peta/images/' + name[0:str_num] + '.jpg')
        height, width = im.size
        if width == 160 and height == 80:
            valid_img.append(name)
            label_selected.append(label[image.index(name)])


    valid_img_selected = valid_img

    pic_size = (224, 224)
    channel0 = 2
    channel1 = 0
    channel2 = 1
    size = [1, 3, 224, 224]
    mean_value = [0.485, 0.456, 0.406]
    std_value = [0.229, 0.224, 0.225]
    for i, key in enumerate(valid_img_selected):
        img_path = "../data/input/peta/images/" + key[0:5] + '.jpg'
        img = Image.open(img_path)
        img = img.resize(pic_size, Image.ANTIALIAS)
        img = np.array(img)
        img = img.transpose(channel0, channel1, channel2)
        img = img.reshape(size[0], size[1], size[2], size[3])
        # Normalize and standardize the test image
        image = (img - np.min(img)) / (np.max(img) - np.min(img))
        image[0][0] = (image[0][0] - mean_value[0]) / std_value[0]
        image[0][1] = (image[0][1] - mean_value[1]) / std_value[1]
        image[0][2] = (image[0][2] - mean_value[2]) / std_value[2]

        image = image.astype(np.float32)
        image.tofile(os.path.join(args.result_path, key[0:5] + '.bin'))