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
import argparse
import cv2
import torch
import pickle as pk
import numpy as np
from tqdm import tqdm
import mmcv
import multiprocessing

flags = None


def gen_input_bin(file_batches, batch):
    i = 0
    for file in tqdm(file_batches[batch]):
        if file.endswith("jpg"):
            i = i + 1
            image = mmcv.imread(os.path.join(flags.image_src_path, file))
            # ori_shape = image.shape
            image, scalar = mmcv.imrescale(image, (flags.model_input_height, flags.model_input_width), return_scale=True)
            # img_shape = image.shape
            image = mmcv.impad(image, shape=(flags.model_input_height, flags.model_input_width),
                               pad_val=(flags.model_pad_val, flags.model_pad_val, flags.model_pad_val))

            #image = image.transpose(2, 0, 1)
            image = image.astype(np.uint8)
            image.tofile(os.path.join(flags.bin_file_path, file.split('.')[0] + ".bin"))
            image_meta = {'scalar': scalar}
            with open(os.path.join(flags.meta_file_path, file.split('.')[0] + ".pk"), "wb") as fp:
                pk.dump(image_meta, fp)


def preprocess():
    files = os.listdir(flags.image_src_path)
    file_batches = [files[i:i + 100] for i in range(0, len(files), 100) if files[i:i + 100] != []]
    thread_pool = multiprocessing.Pool(len(file_batches))
    for batch in range(len(file_batches)):
        thread_pool.apply_async(gen_input_bin, args=(file_batches, batch))
    thread_pool.close()
    thread_pool.join()
    print("in thread, except will not report! please ensure bin files generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocess of YOLOX PyTorch model')
    parser.add_argument("--image_src_path", default="/opt/npu/coco/val2017", help='image of dataset')
    parser.add_argument("--bin_file_path", default="val2017_bin", help='Preprocessed image buffer')
    parser.add_argument("--meta_file_path", default="val2017_bin_meta", help='Get image meta')
    parser.add_argument("--model_input_height", default=640, type=int, help='input tensor height')
    parser.add_argument("--model_input_width", default=640, type=int, help='input tensor width')
    parser.add_argument("--model_pad_val", default=114, type=int, help='image pad value')
    flags = parser.parse_args()
    if not os.path.exists(flags.bin_file_path):
        os.makedirs(flags.bin_file_path)
    if not os.path.exists(flags.meta_file_path):
        os.makedirs(flags.meta_file_path)
    preprocess()
