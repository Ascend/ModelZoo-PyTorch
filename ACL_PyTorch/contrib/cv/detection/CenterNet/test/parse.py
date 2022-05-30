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


if __name__ == '__main__':
    if sys.argv[1].endswith('.txt'):
        result_txt = sys.argv[1]
        with open(result_txt, 'r') as f:
            content = f.read()
        fps = content.split('\n')[-2].split(',')[1].split(':')[-1] # 解析310P的结果
        # fps = float(content.split('\n')[-2].split(',')[1].split(':')[-1]) * 解析310的结果
        print('710 bs{} fps:{}'.format(result_txt.split('_')[3], fps))