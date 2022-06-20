# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import re
import sys
import json
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-file', type=str, default="./msame_bs1.txt")
    parser.add_argument('--batch-size', type=int, default=1)
    args = parser.parse_args()

    if args.result_file.endswith('.json'):
        result_json = args.result_file
        with open(result_json, 'r') as f:
            content = f.read()
        tops = [i.get('value') for i in json.loads(content).get('value') if 'Top' in i.get('key')]
        print('om {} top1:{}'.format(result_json.split('_')[1].split('.')[0], tops[0]))
    elif args.result_file.endswith('.txt'):
        result_txt = args.result_file
        with open(result_txt, 'r') as f:
            content = f.read()
        txt_data_list = re.findall(r'Inference average time without first time:.*ms', content.replace('\n', ',') + ',')[-1]
        avg_time = txt_data_list.split(' ')[-2]
        fps = args.batch_size * 1000 / float(avg_time)
        print('310P bs{} fps:{:.3f}'.format(args.batch_size, fps))
