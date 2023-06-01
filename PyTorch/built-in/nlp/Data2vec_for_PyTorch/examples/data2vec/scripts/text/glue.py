# coding:utf-8
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

from valids import parser, main as valids_main
import os.path as osp


args = parser.parse_args()
args.target = "valid_accuracy"
args.best_biggest = True
args.best = True
args.last = 0
args.path_contains = None

res =  valids_main(args, print_output=False)

grouped = {}
for k, v in res.items():
    k = osp.dirname(k)
    run = osp.dirname(k)
    task = osp.basename(k)
    val = v["valid_accuracy"]

    if run not in grouped:
        grouped[run] = {}

    grouped[run][task] = val

for run, tasks in grouped.items():
    print(run)
    avg = sum(float(v) for v in tasks.values()) / len(tasks)
    avg_norte = sum(float(v) for k,v in tasks.items() if k != 'rte') / (len(tasks) -1)
    try:
        print(f"{tasks['cola']}\t{tasks['qnli']}\t{tasks['mrpc']}\t{tasks['rte']}\t{tasks['sst_2']}\t{avg:.2f}\t{avg_norte:.2f}")
    except:
        print(tasks)
    print()
