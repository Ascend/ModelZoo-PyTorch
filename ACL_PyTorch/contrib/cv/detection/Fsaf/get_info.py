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
import cv2
from glob import glob


def get_bin_info(file_path, info_name, width, height):
    bin_images = glob(os.path.join(file_path, '*.bin'))
    with open(info_name, 'w') as file:
        for index, img in enumerate(bin_images):
            content = ' '.join([str(index), img, width, height])
            file.write(content)
            file.write('\n')


def get_jpg_info(file_path, info_name):
    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
    image_names = []
    for extension in extensions:
        image_names.append(glob(os.path.join(file_path, '*.' + extension)))  
    with open(info_name, 'w') as file:
        for image_name in image_names:
            if len(image_name) == 0:
                continue
            else:
                for index, img in enumerate(image_name):
                    img_cv = cv2.imread(img)
                    shape = img_cv.shape
                    width, height = shape[1], shape[0]
                    content = ' '.join([str(index), img, str(width), str(height)])
                    file.write(content)
                    file.write('\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_type", default="")
    parser.add_argument("--file_path", default="")
    parser.add_argument("--info_name", default="")
    args = parser.parse_args()
    get_bin_info(args.file_type, args.file_path, args.info_name)

