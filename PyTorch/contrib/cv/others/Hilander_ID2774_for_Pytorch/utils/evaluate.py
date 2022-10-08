# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
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
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import inspect
import argparse
import numpy as np

from utils import Timer, TextColors, metrics
from clustering_benchmark import ClusteringBenchmark

def _read_meta(fn):
    labels = list()
    lb_set = set()
    with open(fn) as f:
        for lb in f.readlines():
            lb = int(lb.strip())
            labels.append(lb)
            lb_set.add(lb)
    return np.array(labels), lb_set


def evaluate(gt_labels, pred_labels, metric='pairwise'):
    if isinstance(gt_labels, str) and isinstance(pred_labels, str):
        print('[gt_labels] {}'.format(gt_labels))
        print('[pred_labels] {}'.format(pred_labels))
        gt_labels, gt_lb_set = _read_meta(gt_labels)
        pred_labels, pred_lb_set = _read_meta(pred_labels)

        print('#inst: gt({}) vs pred({})'.format(len(gt_labels),
                                                 len(pred_labels)))
        print('#cls: gt({}) vs pred({})'.format(len(gt_lb_set),
                                                len(pred_lb_set)))

    metric_func = metrics.__dict__[metric]

    with Timer('evaluate with {}{}{}'.format(TextColors.FATAL, metric,
                                             TextColors.ENDC)):
        result = metric_func(gt_labels, pred_labels)
    if isinstance(result, np.float):
        print('{}{}: {:.4f}{}'.format(TextColors.OKGREEN, metric, result,
                                      TextColors.ENDC))
    else:
        ave_pre, ave_rec, fscore = result
        print('{}ave_pre: {:.4f}, ave_rec: {:.4f}, fscore: {:.4f}{}'.format(
            TextColors.OKGREEN, ave_pre, ave_rec, fscore, TextColors.ENDC))

def evaluation(pred_labels, labels, metrics):
    print('==> evaluation')
    #pred_labels = g.ndata['pred_labels'].cpu().numpy()
    max_cluster = np.max(pred_labels)
    #gt_labels_all = g.ndata['labels'].cpu().numpy()
    gt_labels_all = labels
    pred_labels_all = pred_labels
    metric_list = metrics.split(',')
    for metric in metric_list:
        evaluate(gt_labels_all, pred_labels_all, metric)
    # H and C-scores
    gt_dict = {}
    pred_dict = {}
    for i in range(len(gt_labels_all)):
        gt_dict[str(i)] = gt_labels_all[i]
        pred_dict[str(i)] = pred_labels_all[i]
    bm = ClusteringBenchmark(gt_dict)
    scores = bm.evaluate_vmeasure(pred_dict)
    fmi_scores = bm.evaluate_fowlkes_mallows_score(pred_dict)
    print(scores)
