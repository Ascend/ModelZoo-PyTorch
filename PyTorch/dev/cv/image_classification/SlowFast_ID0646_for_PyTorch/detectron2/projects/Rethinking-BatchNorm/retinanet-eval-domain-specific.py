#!/usr/bin/env python
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
# Copyright (c) Facebook, Inc. and its affiliates.
import sys
import torch
from fvcore.nn.precise_bn import update_bn_stats
from torch import nn

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.evaluation import inference_on_dataset
from detectron2.utils.events import EventStorage
from detectron2.utils.logger import setup_logger

logger = setup_logger()
setup_logger(name="fvcore")


class CycleBatchNormList(nn.ModuleList):
    """
    A hacky way to implement domain-specific BatchNorm
    if it's guaranteed that a fixed number of domains will be
    called with fixed order.
    """

    def __init__(self, length, channels):
        super().__init__([nn.BatchNorm2d(channels, affine=False) for k in range(length)])
        # shared affine, domain-specific BN
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self._pos = 0

    def forward(self, x):
        ret = self[self._pos](x)
        self._pos = (self._pos + 1) % len(self)

        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        return ret * w + b


if __name__ == "__main__":
    checkpoint = sys.argv[1]
    cfg = LazyConfig.load_rel("./configs/retinanet_SyncBNhead.py")
    model = cfg.model
    model.head.norm = lambda c: CycleBatchNormList(len(model.head_in_features), c)
    model = instantiate(model)
    model.cuda()
    DetectionCheckpointer(model).load(checkpoint)

    cfg.dataloader.train.total_batch_size = 8
    logger.info("Running PreciseBN ...")
    with EventStorage(), torch.no_grad():
        update_bn_stats(model, instantiate(cfg.dataloader.train), 500)

    logger.info("Running evaluation ...")
    inference_on_dataset(
        model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
    )
