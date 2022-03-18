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
import shutil

if __name__ == '__main__':
    bin_path = sys.argv[1]
    resule_path = sys.argv[2]
    if not os.path.exists(resule_path):
        os.mkdir(resule_path)
    f = os.listdir(bin_path)
    for data in f:
        data = data.strip('\n')
        dir_name = data.split('_')[0] + '--' + data.split('_')[1]
        dir_path = os.path.join(resule_path, dir_name)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
    file_list = os.listdir(resule_path)
    for dir in file_list:
        dir = dir.strip('\n')
        cur_path = os.path.join(resule_path, dir)
        for data in f:
            data = data.strip('\n')
            if data.split('_')[0] == dir.split('--')[0]:
                shutil.copy(os.path.join(bin_path, data),
                            os.path.join(cur_path, data))
