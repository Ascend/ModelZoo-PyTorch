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
"""get info"""

import os
import sys
import cv2
from glob import glob


def get_bin_info(bin_file_path, bin_info_name, bin_width, bin_height):
    """get_bin_info"""
    bin_images = glob(os.path.join(bin_file_path, '*.bin'))
    with open(bin_info_name, 'w') as info_file:
        for index, img in enumerate(bin_images):
            content = ' '.join([str(index), img, bin_width, bin_height])
            info_file.write(content)
            info_file.write('\n')


def get_jpg_info(jpg_file_path, jpg_info_name):
    """get_jpg_info"""
    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
    image_names = []
    for extension in extensions:
        image_names.append(glob(os.path.join(jpg_file_path, '*.' + extension)))
    with open(jpg_info_name, 'w') as jpg_file:
        for image_name in image_names:
            if len(image_name) == 0:
                continue
            else:
                for index, img in enumerate(image_name):
                    img_cv = cv2.imread(img)
                    shape = img_cv.shape
                    jpg_width, jpg_height = shape[1], shape[0]
                    content = ' '.join([str(index), img, str(jpg_width), str(jpg_height)])
                    jpg_file.write(content)
                    jpg_file.write('\n')


if __name__ == '__main__':
    file_type = sys.argv[1]
    file_path = sys.argv[2]
    info_name = sys.argv[3]
    if file_type == 'bin':
        width = sys.argv[4]
        height = sys.argv[5]
        assert len(sys.argv) == 6, 'The number of input parameters must be equal to 5'
        get_bin_info(file_path, info_name, width, height)
    elif file_type == 'jpg':
        assert len(sys.argv) == 4, 'The number of input parameters must be equal to 3'
        get_jpg_info(file_path, info_name)