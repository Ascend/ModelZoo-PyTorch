# -*- coding: utf-8 -*-
# Copyright (c) Soumith Chintala 2016,
# All rights reserved
#
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://spdx.org/licenses/BSD-3-Clause.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License


import glob
import os
import sys
from glob import glob

image_names = []
image_size = []

def get_bin_info(file_path, info_name, width, height):
    bin_images = glob(os.path.join(file_path, '*.bin'))
    with open(info_name, 'w') as fp:
        for idx, img in enumerate(bin_images):
            content = ' '.join([str(idx), img, width, height])
            fp.write(content)
            fp.write('\n')


if __name__ == '__main__':
    file_type = sys.argv[1]
    file_path = sys.argv[2]
    info_name = sys.argv[3]
    
    width = sys.argv[4]
    height = sys.argv[5]
    assert len(sys.argv) >= 6, 'The number of input parameters must be equal to 5'
    get_bin_info(file_path, info_name, width, height)
    print('Bin info generated.')
