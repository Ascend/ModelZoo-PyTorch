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
from glob import glob

file_path = sys.argv[1]
info_name = sys.argv[2]
width = sys.argv[3]
height = sys.argv[4]

bin_images = glob(os.path.join(file_path, '*'))

with open(info_name, 'w') as  file:
    for index, img in enumerate(bin_images):
        content = ' '.join([str(index), img, width, height])
        file.write(content)
        file.write('\n')
