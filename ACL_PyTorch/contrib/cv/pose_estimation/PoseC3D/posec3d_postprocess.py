# Copyright 2022 Huawei Technologies Co., Ltd
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
from collections import OrderedDict

from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from mmaction.core import top_k_accuracy


def parse_args():
    parser = argparse.ArgumentParser(
        description='Dataset K400 Postprocessing')
    parser.add_argument('--infer_results', type=str,
                        help='directory of inference results.')
    parser.add_argument('--label_file', type=str,
                        help='path to label file after preprocess.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # load info file
    gt_labels = []
    with open(args.label_file, 'r') as f:
        for i, line in enumerate(f):
            bin_name, label = line.strip().split()
            assert int(bin_name) == i
            gt_labels.append(int(label))

    # load inference result
    result_file_list = os.listdir(args.infer_results)
    result_file_list.sort()
    results = []
    for i, file in enumerate(tqdm(result_file_list)):
        assert int(file.split('_')[0]) == i
        file_path = os.path.join(args.infer_results, file)
        result = np.fromfile(file_path, dtype=np.float32).reshape(20, 51)
        result = torch.from_numpy(result)
        result = result.view(1, 20, -1)
        result = F.softmax(result, dim=2).mean(dim=1).numpy()
        results.extend(result)

    # evaluate
    metric_options = dict(top_k_accuracy=dict(topk=(1, 5)))
    metric = 'top_k_accuracy'
    print(f'Evaluating {metric} ...')
    topk = metric_options.setdefault('top_k_accuracy',
                                     {}).setdefault('topk', (1, 5))
    if isinstance(topk, int):
        topk = (topk, )
    top_k_acc = top_k_accuracy(results, gt_labels, topk)
    for k, acc in zip(topk, top_k_acc):
        print(f'top{k}_acc\t{acc:.4f}')


if __name__ == '__main__':
    main()
