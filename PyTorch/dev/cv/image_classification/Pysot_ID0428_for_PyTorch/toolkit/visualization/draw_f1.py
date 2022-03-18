#
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
#

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import rc
from .draw_utils import COLOR, LINE_STYLE

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

def draw_f1(result, bold_name=None):
    # drawing f1 contour
    fig, ax = plt.subplots()
    for f1 in np.arange(0.1, 1, 0.1):
        recall = np.arange(f1, 1+0.01, 0.01)
        precision = f1 * recall / (2 * recall - f1)
        ax.plot(recall, precision, color=[0,1,0], linestyle='-', linewidth=0.5)
        ax.plot(precision, recall, color=[0,1,0], linestyle='-', linewidth=0.5)
    ax.grid(b=True)
    ax.set_aspect(1)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.axis([0, 1, 0, 1])
    plt.title(r'\textbf{VOT2018-LT Precision vs Recall}')

    # draw result line
    all_precision = {}
    all_recall = {}
    best_f1 = {}
    best_idx = {}
    for tracker_name, ret in result.items():
        precision = np.mean(list(ret['precision'].values()), axis=0)
        recall = np.mean(list(ret['recall'].values()), axis=0)
        f1 = 2 * precision * recall / (precision + recall)
        max_idx = np.argmax(f1)
        all_precision[tracker_name] = precision
        all_recall[tracker_name] = recall
        best_f1[tracker_name] = f1[max_idx]
        best_idx[tracker_name] = max_idx

    for idx, (tracker_name, best_f1) in \
            enumerate(sorted(best_f1.items(), key=lambda x:x[1], reverse=True)):
        if tracker_name == bold_name:
            label = r"\textbf{[%.3f] Ours}" % (best_f1)
        else:
            label = "[%.3f] " % (best_f1) + tracker_name
        recall = all_recall[tracker_name][:-1]
        precision = all_precision[tracker_name][:-1]
        ax.plot(recall, precision, color=COLOR[idx], linestyle='-',
                label=label)
        f1_idx = best_idx[tracker_name]
        ax.plot(recall[f1_idx], precision[f1_idx], color=[0,0,0], marker='o',
                markerfacecolor=COLOR[idx], markersize=5)
    ax.legend(loc='lower right', labelspacing=0.2)
    plt.xticks(np.arange(0, 1+0.1, 0.1))
    plt.yticks(np.arange(0, 1+0.1, 0.1))
    plt.show()

if __name__ == '__main__':
    draw_f1(None)
