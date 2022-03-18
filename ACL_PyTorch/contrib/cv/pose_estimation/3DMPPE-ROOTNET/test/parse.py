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
    if sys.argv[1].endswith('.json'):
        result_json = sys.argv[1]
        with open(result_json, 'r') as f:
            content = f.read()
        result_json = result_json.split('_')[1]
        result_json = result_json.split('/')[0]
        print('om {} accuracy {}'.format(result_json, content))
    elif sys.argv[1].endswith('.txt'):
        result_txt = sys.argv[1]
        with open(result_txt, 'r') as f:
            content = f.read()
        txt_data_list = content.split(' ')
        fps = float(txt_data_list[2].replace('samples/s,', '')) * 4
        print('310 {} fps:{}'.format(result_txt.split('_')[4], fps))