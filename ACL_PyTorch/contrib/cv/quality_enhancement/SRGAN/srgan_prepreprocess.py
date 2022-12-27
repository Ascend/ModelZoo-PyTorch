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

from tqdm import tqdm
from PIL import Image


def prepreprocess(src_path, result_path, upscale_factor):
    src_path_list = os.listdir(src_path)
    for file_name in tqdm(src_path_list):
        image = Image.open(os.path.join(src_path, file_name)).convert('RGB')
        image_w, image_h = image.size
        image_size = (int(image_w / upscale_factor), int(image_h / upscale_factor))
        image = image.resize(image_size, Image.BICUBIC)
        image.save(os.path.join(result_path, file_name))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='prepreprocess script')
    parser.add_argument('--src_path', default='./datasets/Set5/', type=str,
                    help='path of source image files')
    parser.add_argument('--result_path', default='./datasets/Set5_X2/', type=str,
                    help='path of output ')
    parser.add_argument('--upscale_factor', default=2, type=int,
                    help='upscale_factor')
    args = parser.parse_args()

    if not os.path.exists(args.src_path):
        print("---", "ERROR! the input path is not exist", "---")
        exit(-1)
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    prepreprocess(args.src_path, args.result_path, args.upscale_factor)
    