# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import shutil
import math
import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--img_path', type=str)
parser.add_argument("--target_path", type=str)
opts = parser.parse_args()

target_path = opts.target_path

os.makedirs(target_path, exist_ok=True)

tag_dirs = os.listdir(opts.img_path)

for tag_dir in tag_dirs:
    attribute_dirs = os.listdir(os.path.join(opts.imgs, tag_dir))
    for attribute_dir in attribute_dirs:
        open(os.path.join(target_path, f'{tag_dir}_{attribute_dir}.txt'), 'w')
        images = os.listdir(os.path.join(opts.imgs, tag_dir, attribute_dir))
        for image in images:
            if os.path.isfile(image):
                with open(os.path.join(target_path, f'{tag_dir}_{attribute_dir}.txt'), mode='a') as f:
                    f.write(f'{os.path.abspath(os.path.join(opts.imgs, tag_dir, attribute_dir, image))} 0\n')
