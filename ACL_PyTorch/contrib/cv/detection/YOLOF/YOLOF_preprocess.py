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
import argparse
import numpy as np
from PIL import Image
import pickle as pk
import multiprocessing

flags = None
width = 608
height = 608
pixel_mean = np.array([103.5300, 116.2800, 123.6750], dtype=np.float32)


def gen_input_bin(file_batches, batch):
    i = 0
    image_bag = []
    image_meta_bag = []
    for file in file_batches[batch]:
        i = i + 1
        src = Image.open(os.path.join(flags.image_src_path, file)).convert("RGB")
        ori_shape = (src.size[1], src.size[0])
        image = src.resize((height, width), 2)
        image = np.asarray(image)
        image = image[..., ::-1]
        image = image - pixel_mean
        image = image.transpose(2, 0, 1)
        image_meta = {'ori_shape': ori_shape, 'file_name': file.split('.')[0]}
        image_bag.append(image)
        image_meta_bag.append(image_meta)
        if len(image_bag) % flags.batch_size == 0:
            np.array(image_bag).tofile(os.path.join(flags.bin_file_path, "{}_{}.bin".format(batch, i)))
            with open(os.path.join(flags.meta_file_path, "{}_{}.pk".format(batch, i)), "wb") as fp:
                pk.dump(image_meta_bag, fp)
            image_bag = []
            image_meta_bag = []
    if image_bag:
        ext_img_bag = [image_bag[-1] for i in range(flags.batch_size-len(image_bag))]
        image_bag += ext_img_bag
        ext_img_meta_bag = [image_meta_bag[-1] for i in range(flags.batch_size - len(image_meta_bag))]
        image_meta_bag += ext_img_meta_bag
        np.array(image_bag).tofile(os.path.join(flags.bin_file_path, "{}_{}.bin".format(batch, i)))
        with open(os.path.join(flags.meta_file_path, "{}_{}.pk".format(batch, i)), "wb") as fp:
            pk.dump(image_meta_bag, fp)


def preprocess():
    step = 100 - (100 % flags.batch_size)
    files = os.listdir(flags.image_src_path)
    file_batches = [files[i:i + step] for i in range(0, 5000, step) if files[i:i + step] != []]
    thread_pool = multiprocessing.Pool(len(file_batches))
    for batch in range(len(file_batches)):
        thread_pool.apply_async(gen_input_bin, args=(file_batches, batch))
    thread_pool.close()
    thread_pool.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocess of YOLOF PyTorch model')
    parser.add_argument("--image_src_path", default="YOLOF/datasets/coco/val2017", help='image of dataset')
    parser.add_argument("--bin_file_path", default="val2017_bin")
    parser.add_argument("--meta_file_path", default="val2017_bin_meta")
    parser.add_argument("--batch_size", default=1, type=int)
    flags = parser.parse_args()
    if not os.path.exists(flags.bin_file_path):
        os.makedirs(flags.bin_file_path)
    if not os.path.exists(flags.meta_file_path):
        os.makedirs(flags.meta_file_path)
    preprocess()
