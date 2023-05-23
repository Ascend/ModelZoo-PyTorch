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

import argparse
import os
import shutil

from tqdm import tqdm

from default_arguments import TEST_IMAGE_DIR, LABEL_FILE, TOTAL_IMAGE_DIR


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-label', type=str, default=LABEL_FILE)
    parser.add_argument('--total-image-dir', type=str, default=TOTAL_IMAGE_DIR)
    parser.add_argument('--test-image-dir', type=str, default=TEST_IMAGE_DIR)
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
