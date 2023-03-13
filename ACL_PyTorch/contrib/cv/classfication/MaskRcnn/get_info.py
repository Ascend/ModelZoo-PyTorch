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

import argparse
import os
import cv2
from glob import glob
from tqdm import tqdm

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
        for image_name in tqdm(image_names):
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
    parser.add_argument("--file_type", default="./origin_pictures.info")
    parser.add_argument("--file_path", default="./result/dumpOutput_device0/")
    parser.add_argument("--info_name", default="./detection-results/")
    parser.add_argument("--width", type=str, default=4)
    parser.add_argument("--height", type=str, default=1344)
    flags = parser.parse_args()

    file_type = flags.file_type
    file_path = flags.file_path
    info_name = flags.info_name
    if file_type == 'bin':
        width = flags.width
        height = flags.height
        get_bin_info(file_path, info_name, width, height)
    elif file_type == 'jpg':
        get_jpg_info(file_path, info_name)