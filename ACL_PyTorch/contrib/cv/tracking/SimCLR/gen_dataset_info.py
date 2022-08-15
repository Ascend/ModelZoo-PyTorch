"""
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
"""
import os
import sys


def get_bin_info(file_path, info_name, width, height):
    """generate bin file info to a file"""
    bin_images = file_path
    with open(info_name, 'w') as file:
        files = os.listdir(bin_images)
        for index, img in enumerate(files):
            content = ' '.join([str(index), bin_images + "/" + img, width, height])
            file.write(content)
            file.write('\n')

if __name__ == '__main__':
    file_type = sys.argv[1]
    data_path = sys.argv[2]
    file_name = sys.argv[3]
    if file_type == 'bin':
        file_width = sys.argv[4]
        file_height = sys.argv[5]
        assert len(sys.argv) == 6, 'The number of input parameters must be equal to 5'
        get_bin_info(data_path, file_name, file_width, file_height)

