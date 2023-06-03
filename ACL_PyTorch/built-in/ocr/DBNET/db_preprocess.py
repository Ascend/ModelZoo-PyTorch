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

import os
import argparse
import multiprocessing

import numpy as np
import cv2

def gen_input_npy(file_batches, batch, save_path):
    i = 0
    for file in file_batches[batch]:
        i = i + 1
        print("batch", batch, file, "===", i)

        image = cv2.imread(os.path.join(flags.image_src_path, file), cv2.IMREAD_COLOR).astype('float32')
        image = cv2.resize(image, (1280, 736)) 
        image -= np.array([122.67891434, 116.66876762, 104.00698793]) # mean values
        image = image / 255.
        image = image.transpose(2, 0, 1)
        image = image[np.newaxis,:] # CHW ---> NCHW
        np.save(os.path.join(save_path, file.split('.')[0] + ".npy"), image)

def preprocess(src_path, save_path):
    files = os.listdir(src_path)
    file_batches = [files[i:i + 100] for i in range(0, 5000, 100) if files[i:i + 100] != []]
    thread_pool = multiprocessing.Pool(len(file_batches))
    for batch in range(len(file_batches)):
        thread_pool.apply_async(gen_input_npy, args=(file_batches, batch, save_path))
    thread_pool.close()
    thread_pool.join()
    print("in thread, except will not report! please ensure npy files generated.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='preprocess of db pytorch')
    parser.add_argument('--image_src_path', default="./datasets/icdar2015/test_images", help='images of dataset')
    parser.add_argument('--npu_file_path', default="./icdar2015_npy/", help='npy data')
    flags = parser.parse_args()
    if not os.path.isdir(flags.npu_file_path):
        os.makedirs(os.path.realpath(flags.npu_file_path))
    preprocess(flags.image_src_path, flags.npu_file_path)
