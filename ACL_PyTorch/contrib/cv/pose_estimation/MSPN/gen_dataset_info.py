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
    bin_info = glob(os.path.join(file_path, '*.bin'))
    with open(info_name, 'w') as file:
        for index, info in enumerate(bin_info):
            content = ' '.join([str(index), info, width, height])
            file.write(content)
            file.write('\n')




if __name__ == '__main__':
    file_path = sys.argv[1]
    info_name = sys.argv[2]
    width =sys.argv[3]
    height =sys.argv[4]
    get_bin_info(file_path, info_name, width, height)
