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

def get_acc(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        last_line = lines[-1]
    psnr = last_line.split(" ")[2]
    print(filename.split('.')[0],"Average PSNR:", psnr)


def get_perf(filename):
    with open(filename, 'r') as f:
        content = f.read()
    txt_data_list = [i.strip() for i in re.findall(r':(.*?),', content.replace('\n', ',') + ',')]
    fps = 1000/float((txt_data_list[2].split(' '))[0]) * 4
    print('310  fps:{}'.format(fps))
   
if __name__ == "__main__":

    filename = sys.argv[1]

    if filename.endswith(".log"):
        get_acc(filename)
    elif filename.endswith(".txt"):
        get_perf(filename)