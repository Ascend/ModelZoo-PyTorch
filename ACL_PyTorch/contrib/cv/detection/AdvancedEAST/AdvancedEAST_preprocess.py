# Copyright 2021 Huawei Technologies Co., Ltd
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

import sys
import os
import multiprocessing
import math
import numpy as np
from PIL import Image

sys.path.append('./AdvancedEAST-PyTorch')

from preprocess import reorder_vertexes


train_task_id = '3T736'
max_train_img_size = int(train_task_id[-3:])
validation_split_ratio = 0.1
data_dir = 'icpr'
bin_dir = 'prep_dataset'
origin_image_dir = os.path.join(data_dir, 'image_10000')
origin_txt_dir = os.path.join(data_dir, 'txt_10000')
train_image_dir = os.path.join(data_dir, 'images_{}/'.format(train_task_id))
train_label_dir = os.path.join(data_dir, 'labels_{}/'.format(train_task_id))


def gen_data(img_list):
    for img_fname in img_list:
        with Image.open(os.path.join(origin_image_dir, img_fname)) as im:  # 打开每张图片
            d_wight, d_height = max_train_img_size, max_train_img_size
            scale_ratio_w = d_wight / im.width
            scale_ratio_h = d_height / im.height
            im = im.resize((d_wight, d_height), Image.NEAREST).convert('RGB')  # 图片缩放
            with open(os.path.join(origin_txt_dir, img_fname[:-4] + '.txt'), 'r', encoding='UTF-8') as f:
                anno_list = f.readlines()
            xy_list_array = np.zeros((len(anno_list), 4, 2))
            for anno, i in zip(anno_list, range(len(anno_list))):
                anno_colums = anno.strip().split(',')
                anno_array = np.array(anno_colums)
                xy_list = np.reshape(anno_array[:8].astype(float), (4, 2))
                xy_list[:, 0] = xy_list[:, 0] * scale_ratio_w  # 坐标缩放
                xy_list[:, 1] = xy_list[:, 1] * scale_ratio_h
                xy_list = reorder_vertexes(xy_list)  # 坐标顺序转换为统一格式
                xy_list_array[i] = xy_list
            
            img_fname = img_fname[:-4].replace('.', '-') + img_fname[-4:]
            im.save(os.path.join(train_image_dir, img_fname))
            np.save(os.path.join(
                train_label_dir,
                img_fname[:-4] + '.npy'),
                xy_list_array)  # 保存顺序一致处理后的坐标点集
            
            img = Image.open(os.path.join(train_image_dir, img_fname)).convert('RGB')
            img = np.array(img).astype(np.float32) / 255
            img = np.transpose(img, (2, 0, 1))
            img.tofile(os.path.join(bin_dir, img_fname[:-4] + '.bin'))


def preprocess():
    o_img_list = os.listdir(origin_image_dir)
    print('found {} origin images.'.format(len(o_img_list)))
    o_img_list.sort()
    val_count = int(validation_split_ratio * len(o_img_list))
    val_img_list = o_img_list[:val_count]

    workers = multiprocessing.cpu_count()
    batch_size = math.ceil(len(o_img_list) / workers)
    batch_list = [val_img_list[i * batch_size:(i + 1) * batch_size] for i in range(workers)]
    thread_pool = multiprocessing.Pool(workers)
    for i in range(workers):
        thread_pool.apply_async(gen_data, args=(batch_list[i], ))
    thread_pool.close()
    thread_pool.join()


if __name__ == '__main__':
    data_dir = sys.argv[1]
    bin_dir = sys.argv[2]
    origin_image_dir = os.path.join(data_dir, 'image_10000')
    origin_txt_dir = os.path.join(data_dir, 'txt_10000')
    train_image_dir = os.path.join(data_dir, 'images_{}/'.format(train_task_id))
    train_label_dir = os.path.join(data_dir, 'labels_{}/'.format(train_task_id))
    if not os.path.exists(train_image_dir):
        os.mkdir(train_image_dir)
    if not os.path.exists(train_label_dir):
        os.mkdir(train_label_dir)
    if not os.path.exists(bin_dir):
        os.mkdir(bin_dir)
    preprocess()
