# Copyright (c) OpenMMLab. All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================

import os
import argparse
import numpy as np
from collections import OrderedDict
from mmaction.core import top_k_accuracy
import torch
import pdb
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser(
        description='Dataset K400 Postprocessing')
    parser.add_argument('--result_path', default='/home/wyy/output/out_bs1/20220414_113751', type=str)
    parser.add_argument('--info_path', default='/home/wyy/data/hmdb51.info', type=str)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # load info file
    gt_labels = []
    with open(args.info_path, 'r') as f:
        for line in f.readlines():
            t = line.split( )[-1]
            gt_labels.append(int(t))

    # load inference result
    results = []

    num_file = len(os.listdir(args.result_path))
    for idx in range(num_file):
        file = os.path.join(args.result_path, str(idx) + '_output_0.txt')
        result = np.loadtxt(file)
        result = torch.from_numpy(result)
        batch_size = result.shape[0]
#        pdb.set_trace()
        result = result.view(batch_size // 20, 20, -1)  # cls_score = cls_score.view(batch_size // num_segs, num_segs, -1)

        result = F.softmax(result, dim=2).mean(dim=1).numpy()   # cls_score = F.softmax(cls_score, dim=2).mean(dim=1)
        results.extend(result)


    metrics = ['top_k_accuracy']
    metric_options = dict(top_k_accuracy=dict(topk=(1, 5)))
    eval_results = OrderedDict()
    for metric in metrics:
        print(f'Evaluating {metric} ...')
        if metric == 'top_k_accuracy':
            topk = metric_options.setdefault('top_k_accuracy',
                                             {}).setdefault('topk', (1, 5))
            if not isinstance(topk, (int, tuple)):
                raise TypeError(
                    f'topk must be int or tuple of int, but got {type(topk)}')
            if isinstance(topk, int):
                topk = (topk, )

            top_k_acc = top_k_accuracy(results, gt_labels, topk)
            log_msg = []
            for k, acc in zip(topk, top_k_acc):
                eval_results[f'top{k}_acc'] = acc
                log_msg.append(f'\ntop{k}_acc\t{acc:.4f}')
            log_msg = ''.join(log_msg)
            print(log_msg)
            continue


if __name__ == '__main__':
    main()