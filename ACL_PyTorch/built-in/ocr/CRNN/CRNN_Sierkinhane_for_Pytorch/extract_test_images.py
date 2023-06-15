# Copyright 2021 Huawei Technologies Co., Ltd
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

# coding=utf-8

import os
import shutil
import sys

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from argument_parser import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_test_label_argument()
    parser.add_total_image_dir_argument()
    parser.add_test_image_dir_argument()
    return parser.parse_args()


def extract_test_images(test_label, total_image_dir, test_image_dir):
    if not os.path.exists(test_image_dir):
        os.mkdir(test_image_dir)
    lines = open(test_label, encoding='utf-8').readlines()
    test_images = [line.split(' ')[0] for line in lines]
    for image in tqdm(test_images):
        shutil.copyfile(
            os.path.join(total_image_dir, image),
            os.path.join(test_image_dir, image)
        )


if __name__ == '__main__':
    args = parse_args()
    extract_test_images(args.test_label, args.total_image_dir, args.test_image_dir)
