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

import sys
import re
if __name__ == '__main__':
    mask_path = sys.argv[1]
    refine_path = sys.argv[2]

    second = 0.0

    with open(mask_path,'r') as f:
        file = f.read()
        f.close()
    useful_str = re.findall(f'mean = (.*?),', file)[0]
    second += float(useful_str)

    with open(refine_path, 'r') as f:
        file = f.read()
        f.close()
    useful_str = re.findall(f'mean = (.*?),', file)[0]
    second += float(useful_str)

    result = 1000/(second/4)
    print(f'310P3 bs{1} fps:{result}')

