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


def get_bin_info(file_type, file_path, info_name, width, height):
    bin_files = sorted(os.listdir(file_path))

    with open(info_name+'.info', 'w') as file:
        i = 0
        for bin_file in bin_files:
            if bin_file.endswith('.bin'):
                if file_type == 'tem':
                    content = ' '.join([str(i), 'output/BSN-TEM-preprocess/feature'+'/'+bin_file, width, height])
                if file_type == 'pem':
                    content = ' '.join([str(i), 'output/BSN-PEM-preprocess/feature'+'/'+bin_file, width, height])
                file.write(content)
                file.write('\n')
                i = i+1

if __name__ == '__main__':
    file_type = sys.argv[1]
    file_path = sys.argv[2]
    info_name = sys.argv[3]
    line = sys.argv[4]
    col = sys.argv[5]
    get_bin_info(file_type,file_path, info_name, line, col)