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

import argparse
import sys
import os
import numpy as np
import json

sys.path.append('./Swin-Transformer')


def parse_option():
    parser = argparse.ArgumentParser('Swin transformer postprocess args helper', add_help=False)
    parser.add_argument('--result_path', type=str, required=True, metavar="FILE", help='path to om results', )
    parser.add_argument('--target_file', type=str, required=True, metavar="FILE", help='path to target file', )
    parser.add_argument('--save_path', type=str, required=True, metavar="FILE", help='path to save predicting result', )
    args, _ = parser.parse_known_args()
    return args


def postprocess(result_path, target_file, save_file):
    re_files = os.listdir(result_path)
    labels = json.load(open(target_file, 'rb'))
    top1_cnt = 0.0
    top5_cnt = 0.0
    for file in re_files:
        result = np.loadtxt(os.path.join(result_path, file))
        img_name = file.split('.')[0]
        ans = labels[img_name]
        if ans == result.argmax():
            top1_cnt = top1_cnt + 1
            top5_cnt = top5_cnt + 1
        else:
            for p in range(5):
                if ans == result.argmax():
                    top5_cnt = top5_cnt + 1
                    break
                result[result.argmax()] = 0
    ans = {}
    ans['Accuracy@1'] = top1_cnt / len(re_files)
    ans['Accuracy@5'] = top5_cnt / len(re_files)
    print(ans)
    writer = open(save_file, 'w')
    json.dump(ans, writer)
    writer.close()


if __name__ == '__main__':
    args = parse_option()
    postprocess(args.result_path, args.target_file, args.save_path)
