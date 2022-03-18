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

import os
import argparse
import numpy as np
import json

def process_pred(pred_file):
    """Get predicted label from predction

    Args:
        pred_file (str): prediction file

    Returns:
        int: predicted label
    """
    data = np.loadtxt(pred_file)
    assert len(data) == 1000
    pred_label = data.argmax()
    return pred_label


def pred_eval(label_file, pred_dir):
    """evaluate predictions

    Args:
        label_file (str): path of groundtruth file
        pred_dir (str): path of predictions
    """
    with open(label_file, 'r') as f:
        gt = json.load(f)
    output_file_list = os.listdir(pred_dir)
    result = []
    for output_file in output_file_list:
        output_name = '_'.join(output_file.split('_')[:3])
        gt_label = gt[output_name]
        pred_label = process_pred(os.path.join(pred_dir, output_file))
        result.append(gt_label == pred_label)
    print('Validation Results for', pred_dir)
    print("Top 1 Accuracy: {:.1%}".format(sum(result) / len(result)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_file", default="/home/Datasets/imagenet/imagenet_labels_fixres.json")
    parser.add_argument("--pred_dir", default="./result/dumpOutput_device0_bs1/")
    args = parser.parse_args()
    pred_eval(args.label_file, args.pred_dir)
