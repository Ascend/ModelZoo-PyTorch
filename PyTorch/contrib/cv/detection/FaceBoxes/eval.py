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

def compare(num1, ap1, num2, ap2):
    if(ap1 >= ap2):
        return num1, ap1
    else:
        return num2, ap2


if __name__ == '__main__':
    i = 280
    count = 0
    max_num = 0
    max_ap = 0
    logFile = sys.argv[1]
    with open(logFile, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            cur_num = i
            cur_ap = float(line.split(':')[1])
            max_num, max_ap = compare(max_num, max_ap, cur_num, cur_ap)
            count += 1
            if(count < 8):
                i += 5
            elif(count == 8):
                i += 2
            else:
                i += 1
    print(max_num, max_ap)
