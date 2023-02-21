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
import sys
import cv2
from glob import glob


def get_bin_info(file_path, info_name, width, height):
    index = 0
    with open(info_name, 'w') as file:    
        for classes in os.listdir(file_path):
            bin_dir_path = os.path.join(file_path, classes)
            bin_images = glob(os.path.join(bin_dir_path, '*.bin'))
            for img in bin_images:
                content = ' '.join([str(index), img, width, height])
                file.write(content)
                file.write('\n')
                index = index + 1


if __name__ == '__main__':
    file_type = sys.argv[1]
    file_path = sys.argv[2]
    info_name = sys.argv[3]
    assert file_type == 'bin', 'The file_type must is bin'
    if file_type == 'bin':
        width = sys.argv[4]
        height = sys.argv[5]
        assert len(sys.argv) == 6, 'The number of input parameters must be equal to 5'
        get_bin_info(file_path, info_name, width, height)