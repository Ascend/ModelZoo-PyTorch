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

file_path = sys.argv[1]
coco_info = sys.argv[2]
info_name = sys.argv[3]

image_names = []
image_size = []

with open(coco_info, 'r') as file:
    contents = file.read().split('\n')

for content in contents[:-1]:
    temp = content.split()
    key = temp[1]
    image_names.append(key[key.rfind('/') + 1:].split('.')[0])
    image_size.append([temp[2], temp[3]])

name_size = dict(zip(image_names, image_size))

with open(info_name, 'w') as  file:
    index = 0
    for key, val in name_size.items():
        bin_name = os.path.join(file_path, '{}.bin'.format(key))
        content = ' '.join([str(index), bin_name, val[0], val[1]])
        file.write(content)
        file.write('\n')
        index += 1


