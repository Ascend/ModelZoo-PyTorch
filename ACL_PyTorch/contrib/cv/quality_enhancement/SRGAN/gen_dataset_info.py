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
import cv2
from glob import glob
import argparse

parser = argparse.ArgumentParser(description='SRGAN get_info script')
parser.add_argument('--file_type', default='bin', type=str)
parser.add_argument('--file_path', default='./preprocess_data', type=str)
parser.add_argument('--info_name', default='./img.info', type=str)
args = parser.parse_args()


def get_bin_info(file_path, info_name, width, height):
    bin_images = ['./bin/' + x for x in os.listdir(file_path) if x.endswith('.bin')]
    # bin_images = glob(os.path.join(file_path, '*.bin'))
    with open(info_name, 'w') as file:
        for index, img in enumerate(bin_images):
            content = ' '.join([str(index), img, width, height])
            file.write(content)
            file.write('\n')

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def get_jpg_info(file_path, info_name):
    """[get jpg info]

    Args:
        file_path: [file path]
        info_name: [info name]
    """
    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG','png']
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

def get_file_info(file_type, file_path, info_name):
    # 获取文件夹的列表
    dir_list = [x for x in os.listdir(file_path) if os.path.isdir(os.path.join(file_path, x) )]
    for dir in dir_list:
        width = dir.split('_')[1]
        height = dir.split('_')[2]
        dir_path = os.path.join(file_path,dir)
        info_path = os.path.join(file_path,dir,info_name)
        if file_type == "bin":
            get_bin_info(os.path.join(dir_path,'bin'), info_path, width, height)
        elif file_type == 'jpg':
            get_jpg_info(os.path.join(dir_path,'png'), info_path)


if __name__ == '__main__':
    get_file_info(file_type=args.file_type, file_path=args.file_path, info_name=args.info_name)
