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
import os
import argparse

import numpy as np

def postprocess(output_dir, label_path):
    total = 0
    correct = 0
    with open(label_path) as label_file:
        targets = label_file.readline().split()
        i = 0
        label_dict = {}
        while targets:
            label_dict[i] = targets
            targets = label_file.readline().split()
            i += 1

    for f in os.listdir(output_dir):
        output = np.fromfile(os.path.join(output_dir, f), dtype=np.float32).reshape(-1, 1000)
        sorted_index = np.argsort(-output, axis=1)
        data_num = f.split('_')[0]
        targets = label_dict[int(data_num)]
        total += len(targets)
        for i, t in enumerate(targets):
            if int(t) == sorted_index[i][0]:
                correct += 1
    
    print("top 1 accuracy: {}".format(str(correct / total)))

if __name__ == "__main__":
    """
    python3.7 postprocess.py --output_dir=outputs --label_path=label.txt
    """

    parser = argparse.ArgumentParser(description='EfficientNet postprocess')
    parser.add_argument('--output_dir', type=str, help='infer output path', required=True)
    parser.add_argument('--label_path', type=str, help='label.txt path', default='label.txt')
    args = parser.parse_args()

    postprocess(args.output_dir, args.label_path)
    