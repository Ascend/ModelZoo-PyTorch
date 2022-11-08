# -*- coding: utf-8 -*-
# BSD 3-Clause License
#
# Copyright (c) 2017
# All rights reserved.
# Copyright 2022 Huawei Technologies Co., Ltd
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
# ==========================================================================

#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
"""This file is for benchmark data loading process. It can also be used to
refresh the memcached cache. The command line to run this file is:

$ python -m cProfile -o program.prof tools/analysis/benchmark_processing.py
configs/task/method/[config filename]

Note: When debugging, the `workers_per_gpu` in the config should be set to 0
during benchmark.

It use cProfile to record cpu running time and output to program.prof
To visualize cProfile output program.prof, use Snakeviz and run:
$ snakeviz program.prof
"""
import argparse

import mmcv
from mmcv import Config
from mmdet.datasets import build_dataloader

from mmocr.datasets import build_dataset

assert build_dataset is not None


def main():
    parser = argparse.ArgumentParser(description='Benchmark data loading')
    parser.add_argument('config', help='Train config file path.')
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)

    dataset = build_dataset(cfg.data.train)

    # prepare data loaders
    if 'imgs_per_gpu' in cfg.data:
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu

    data_loader = build_dataloader(
        dataset,
        cfg.data.samples_per_gpu,
        cfg.data.workers_per_gpu,
        1,
        dist=False,
        seed=None)

    # Start progress bar after first 5 batches
    prog_bar = mmcv.ProgressBar(
        len(dataset) - 5 * cfg.data.samples_per_gpu, start=False)
    for i, data in enumerate(data_loader):
        if i == 5:
            prog_bar.start()
        for _ in range(len(data['img'])):
            if i < 5:
                continue
            prog_bar.update()


if __name__ == '__main__':
    main()
