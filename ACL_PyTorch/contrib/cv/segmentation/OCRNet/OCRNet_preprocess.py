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
import cv2
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='preprocess for OCRNet mocel')
    parser.add_argument('--src_path', default="cityscapes", help='cityscapes dataset path, including labels and images')
    parser.add_argument('--bin_file_path', default="cityscapes_bin", help='destination path')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--aipp_mode', default=False, type=bool)
    return parser.parse_args()


def read_files(data_path):
    img_src_path = os.path.join(data_path, 'leftImg8bit/val')
    label_src_path = os.path.join(data_path, 'gtFine/val')
    dirs = os.listdir(img_src_path)
    files = []
    for dir in dirs:
        img_list = os.listdir(os.path.join(img_src_path, dir))
        for img_name in img_list:
            label_name = img_name.split('.')[0][:-12] + '_gtFine_labelIds.png'
            files.append({
                "img": os.path.join(img_src_path, dir, img_name),
                "label": os.path.join(label_src_path, dir, label_name),
                "img_name": img_name.split('.')[0],
                "label_name": label_name.split('.')[0],
                "weight": 1
            })
    return files


def convert_label(label, inverse=False):
    ignore_label = 255
    label_mapping = {-1: ignore_label, 0: ignore_label,
                     1: ignore_label, 2: ignore_label,
                     3: ignore_label, 4: ignore_label,
                     5: ignore_label, 6: ignore_label,
                     7: 0, 8: 1, 9: ignore_label,
                     10: ignore_label, 11: 2, 12: 3,
                     13: 4, 14: ignore_label, 15: ignore_label,
                     16: ignore_label, 17: 5, 18: ignore_label,
                     19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                     25: 12, 26: 13, 27: 14, 28: 15,
                     29: ignore_label, 30: ignore_label,
                     31: 16, 32: 17, 33: 18}
    temp = label.copy()
    if inverse:
        for v, k in label_mapping.items():
            label[temp == k] = v
    else:
        for k, v in label_mapping.items():
            label[temp == k] = v
    return label

def gen_sample(image, label, aipp_mode):
    if not aipp_mode:
        image = image.astype(np.float32)[:, :, ::-1]
        image = image / 255.0
        image -= [0.5, 0.5, 0.5]
        image /= [0.5, 0.5, 0.5]
        image = image.transpose((2, 0, 1))
    label = np.array(label).astype('int32')
    return image, label

def gen_input_bin(file_batch_indexes, files, bin_file_path, aipp_mode):

    for i, indexes in enumerate(file_batch_indexes):
        image_bag = []
        label_bag = []
        for id in indexes:
            item = files[id]
            image = cv2.imread(item["img"], cv2.IMREAD_COLOR)

            label = cv2.imread(item["label"], cv2.IMREAD_GRAYSCALE)
            label = convert_label(label)

            image, label = gen_sample(image, label, aipp_mode)
            image_bag.append(image)
            label_bag.append(label)
        # genarate binary file directory
        image_bin_dir = os.path.join(bin_file_path, 'imgs')
        label_bin_dir = os.path.join(bin_file_path, 'labels')
        if not os.path.exists(image_bin_dir):
            os.makedirs(image_bin_dir)
        if not os.path.exists(label_bin_dir):
            os.makedirs(label_bin_dir)
        np.array(image_bag).tofile(os.path.join(image_bin_dir, 'image_bin'+ str(i) + '.bin'))
        np.array(label_bag).tofile(os.path.join(label_bin_dir, 'label_bin' + str(i) + '.bin'))
        print(i+1, "batches are processed...")
    print("finished!")

def main(args):
    data_path = args.src_path
    bin_file_path = args.bin_file_path
    batch_size = args.batch_size
    aipp_mode = args.aipp_mode
    files = read_files(data_path)

    if not os.path.exists(bin_file_path):
        os.makedirs(bin_file_path)

    cur_i = 0
    max_i = batch_size
    file_batch_indexes = []
    while max_i <= len(files):
        cur_batch = [i for i in range(cur_i, max_i)]
        file_batch_indexes.append(cur_batch)
        cur_i = max_i
        max_i = max_i+batch_size

    gen_input_bin(file_batch_indexes, files, bin_file_path, aipp_mode)

if __name__ == '__main__':
    args = parse_args()
    main(args)
