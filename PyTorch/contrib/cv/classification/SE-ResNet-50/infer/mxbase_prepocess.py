# Copyright 2022 Huawei Technologies Co., Ltd
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

import os
import sys
import numpy as np
from sdk.main import prepocess
def imgtobin(file_path, bin_path):
    in_files = os.listdir(file_path)
    if not os.path.exists(bin_path):
        os.makedirs(bin_path)
    i = 0
    for file in in_files:
        i = i + 1
        print(file, "====", i)
        file_dir = os.path.join(file_path, file)
        img = prepocess(file_dir)
        img.tofile(os.path.join(bin_path, file.split('.')[0] + '.bin'))


if __name__ == "__main__":
    file_path = os.path.abspath(sys.argv[1])
    bin_path = os.path.abspath(sys.argv[2])
    imgtobin(file_path, bin_path)
