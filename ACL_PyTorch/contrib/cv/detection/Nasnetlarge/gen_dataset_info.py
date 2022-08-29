"""
Copyright 2020 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import sys
from glob import glob
import cv2


def get_bin_info(file_path, info_name, width, height):
    """get input bin data info"""
    bin_images = glob(os.path.join(file_path, '*.bin'))
    with open(info_name, 'w') as out_file:
        for index, img in enumerate(bin_images):
            content = ' '.join([str(index), img, width, height])
            out_file.write(content)
            out_file.write('\n')


def get_jpg_info(file_path, info_name):
    """get input jpg data info"""
    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
    image_names = []
    for extension in extensions:
        image_names.append(glob(os.path.join(file_path, '*.' + extension)))
    with open(info_name, 'w') as out_file:
        for image_name in image_names:
            if image_name:
                for index, img in enumerate(image_name):
                    img_cv = cv2.imread(img)
                    shape = img_cv.shape
                    img_w, img_h = shape[1], shape[0]
                    content = ' '.join([str(index), img, str(img_w), str(img_h)])
                    out_file.write(content)
                    out_file.write('\n')


if __name__ == '__main__':
    input_type = sys.argv[1]
    input_path = sys.argv[2]
    input_name = sys.argv[3]
    if input_type == 'bin':
        input_width = sys.argv[4]
        input_height = sys.argv[5]
        assert len(sys.argv) == 6, 'The number of input parameters must be equal to 5'
        get_bin_info(input_path, input_name, input_width, input_height)
    elif input_type == 'jpg':
        assert len(sys.argv) == 4, 'The number of input parameters must be equal to 3'
        get_jpg_info(input_path, input_name)
