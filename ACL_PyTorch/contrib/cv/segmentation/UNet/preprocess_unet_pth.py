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
# -*- coding: utf-8 -*-

import time
import shutil
import argparse
import os
import numpy as np
from PIL import Image
import multiprocessing


def gen_bin(files_list, batch, scale=1):
    i = 0
    for file in files_list[batch]:
        i += 1
        print(file, "===", batch, i)

        image = Image.open('{}/{}'.format(src_path, file))

        width, height = image.size
        width_scaled = int(width * scale)
        height_scaled = int(height * scale)
        image_scaled = image.resize((572, 572))
        image_array = np.array(image_scaled, dtype=np.float32)
        image_array = image_array.transpose(2, 0, 1) # HWC -> CHW
        image_array = image_array / 255

        image_array.tofile(os.path.join(save_path, file.split('.')[0] + ".bin"))


def preprocess_images(src_path, save_path):
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
        os.makedirs(save_path)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    files = os.listdir(src_path)
    files_list = [files[i:i + 300] for i in range(0, 5000, 300) if files[i:i + 300] != []]

    st = time.time()
    pool = multiprocessing.Pool(len(files_list))
    for batch in range(len(files_list)):
        pool.apply_async(gen_bin, args=(files_list, batch))
    pool.close()
    pool.join()
    print('Multiple processes executed successfully')
    print('Time Used: {}'.format(time.time() - st))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', default='./carvana/train')
    parser.add_argument('--save_bin_path', default='./prep_bin')
    args = parser.parse_args()

    src_path = args.src_path
    save_path = args.save_bin_path
    preprocess_images(src_path, save_path)
