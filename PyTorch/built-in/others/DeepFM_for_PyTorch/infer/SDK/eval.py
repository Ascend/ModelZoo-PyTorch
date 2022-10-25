# Copyright(C) 2022. Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


"""model evaluation"""
import argparse
from itertools import islice
import numpy as np
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument(
    '--pred_file',
    default='./results.txt',
    help='preditions')
parser.add_argument(
    '--label_file',
    default='../data/val/val.txt',
    help='ground truth')

args = parser.parse_args()
print(f"Loading label_file: {args.label_file}")
vals = np.loadtxt(args.label_file, delimiter='\t',skiprows = 1,dtype = float)
labels = []
for val in vals:
    labels.append(val[0])

print(f"Loading pred_file: {args.pred_file}")
preds_in = open(args.pred_file, 'r')
preds_list = list (islice(preds_in, None))
preds_list = preds_list[0].strip().split('\t')
preds = [];
for pred in preds_list:
    preds.append(float(pred))
auc = roc_auc_score(labels, preds)

print('AUC: ', auc)
