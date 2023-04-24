# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
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
import json
import argparse

import numpy as np


def postprocess(output_path, label_path):
    number = 0
    correct = 0
    label_files = os.listdir(label_path)
    for label_file in label_files:
        labels = np.load(os.path.join(label_path, label_file))
        output_file = label_file[: -4] + '_0.npy'
        predicts = np.load(os.path.join(output_path, output_file))
        for i in range(labels.shape[0]):
            number += 1
            predict = predicts[i]
            label = int(labels[i])
            if predict[label] > predict[1 - label]:
                correct += 1
    print('accuracy:', correct/number)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', required=True,
                        help='path of predict path')
    parser.add_argument('--label_path', required=True,
                        help='path of the folder of labels')
    args = parser.parse_args()

    postprocess(args.output_path, args.label_path)
