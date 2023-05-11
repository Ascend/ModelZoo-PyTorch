# -*- coding: utf-8 -*-

# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# less required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
To generate the evaluation metric AUC.
Usage example: python3 eval.py ./result/test.txt ./result/label.txt
"""

import sys
from numpy import genfromtxt
from sklearn.metrics import roc_auc_score


if __name__ == '__main__':
    try:
        # infer result file path, "./result/infer_result.txt"
        infer_result_file = sys.argv[1]
        # ground truth file path, "./result/ground_truth.txt"
        ground_truth_file = sys.argv[2]
    except IndexError:
        print("Please enter predict result file path | groud truth file path"
              "Such as: python3 eval.py ./result/infer_result.txt ./result/ground_truth.txt")
        exit(1)

    print("loading infer result file: %s" % infer_result_file)
    preds = genfromtxt(infer_result_file, delimiter=',')

    print("loading ground truth file: %s" % ground_truth_file)
    labels = genfromtxt(ground_truth_file, delimiter=',')

    auc = roc_auc_score(labels, preds)
    print('eval auc is %.8f' % auc)
