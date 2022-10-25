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
import mmcv
import multiprocessing


def gen_input_bin(file_batches, batch):
    model_h = args.input_height
    model_w = args.input_width
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float64)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float64)

    for file in file_batches[batch]:
        img = mmcv.imread(os.path.join(args.image_src_path, file))
        scalar_ratio = min(model_h / img.shape[0], model_w / img.shape[1])
        img = mmcv.imrescale(img, scalar_ratio)
        img = mmcv.imnormalize(img, mean, std, to_rgb=True)
        # pad top-left
        img = mmcv.impad(img, shape=(model_h, model_w))
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, 0)
        img.tofile(os.path.join(args.bin_file_path, file.split('.')[0] + '.bin'))


def preprocess():
    files = os.listdir(args.image_src_path)
    batch_size = 100
    file_batches = [files[i:i + batch_size]
                    for i in range(0, len(files), batch_size)]

    thread_pool = multiprocessing.Pool(len(file_batches))
    for batch in range(len(file_batches)):
        thread_pool.apply_async(gen_input_bin, args=(file_batches, batch))
    thread_pool.close()
    thread_pool.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_src_path", default="./data/coco/val2017/",
                        help='image of dataset')
    parser.add_argument("--bin_file_path", default="./val2017_bin/",
                        help='Preprocessed image buffer')
    parser.add_argument("--input_height", default=800,
                        type=int, help='input tensor height')
    parser.add_argument("--input_width", default=1344,
                        type=int, help='input tensor width')
    args = parser.parse_args()

    os.makedirs(args.bin_file_path, exist_ok=True)
    preprocess()