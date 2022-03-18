# Copyright 2020 Huawei Technologies Co., Ltd
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
import sys
import cv2
from glob import glob


def get_bin_info(file_path, info_name, width, height):
    bin_images = glob(os.path.join(file_path, '*.bin'))
    with open(info_name, 'w') as file:
        print(len(bin_images))
        for index, img in enumerate(bin_images):
            content = ' '.join([str(index), img, width, height])
            file.write(content)
            file.write('\n')


def get_png_info(file_path, info_name):
    extensions = ['png']
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
    file_type = sys.argv[1]
    file_path = sys.argv[2]
    info_name = sys.argv[3]
    if file_type == 'bin':
        width = sys.argv[4]
        height = sys.argv[5]
        assert len(sys.argv) == 6, 'The number of input parameters must be equal to 5'
        get_bin_info(file_path, info_name, width, height)
    elif file_type == 'png':
        assert len(sys.argv) == 4, 'The number of input parameters must be equal to 3'
        get_png_info(file_path, info_name)

# python get_info.py bin prep_dataset_200_304 RDN_prep_bin.info 304 200
# python get_info.py png data RDN_prep_bin.info