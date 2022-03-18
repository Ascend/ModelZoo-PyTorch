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
import json
import re

if __name__ == '__main__':
    if sys.argv[1].endswith('.json'):
        result_json = sys.argv[1]
        with open(result_json, 'r') as f:
            content = f.read()
        R1 = [i.get('value') for i in json.loads(content).get('value') if 'R1' in i.get('key')]
        mAP = [i.get('value') for i in json.loads(content).get('value') if 'mAP' in i.get('key')]
        print('om {} R1:{} mAP:{}'.format(result_json.split('_')[1].split('.')[0], R1[0], mAP[0]))
        
    elif sys.argv[1].endswith('.txt'):
        # fps1
        result_txt1 = sys.argv[1]
        with open(result_txt1, 'r') as f:
            content = f.read()
        txt_data_list = [i.strip() for i in re.findall(r':(.*?),', content.replace('\n', ',') + ',')]
        fps1 = float(txt_data_list[7].replace('samples/s', '')) * 4
        # fps2
        result_txt2 = sys.argv[2]
        with open(result_txt2, 'r') as f:
            content = f.read()
        txt_data_list = [i.strip() for i in re.findall(r':(.*?),', content.replace('\n', ',') + ',')]
        fps2 = float(txt_data_list[7].replace('samples/s', '')) * 4
        
        print('310 bs{} fps:{}'.format(result_txt2.split('_')[3], (fps1 + fps2 ) / 2))