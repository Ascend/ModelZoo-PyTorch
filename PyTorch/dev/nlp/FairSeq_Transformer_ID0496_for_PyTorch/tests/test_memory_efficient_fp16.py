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
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import unittest

import torch

from fairseq.optim.adam import FairseqAdam
from fairseq.optim.fp16_optimizer import MemoryEfficientFP16Optimizer


class TestMemoryEfficientFP16(unittest.TestCase):

    def test_load_state_dict(self):
        # define simple FP16 model
        model = torch.nn.Linear(5, 5).cuda().half()
        params = list(model.parameters())

        # initialize memory efficient FP16 optimizer
        optimizer = FairseqAdam(
            argparse.Namespace(
                lr=[0.00001],
                adam_betas='(0.9, 0.999)',
                adam_eps=1e-8,
                weight_decay=0.0,
            ),
            params,
        )
        me_optimizer = MemoryEfficientFP16Optimizer(
            argparse.Namespace(
                fp16_init_scale=1,
                fp16_scale_window=1,
                fp16_scale_tolerance=1,
                threshold_loss_scale=1,
                min_loss_scale=1e-4,
            ),
            params,
            optimizer,
        )

        # optimizer state is created in the first step
        loss = model(torch.rand(5).cuda().half()).sum()
        me_optimizer.backward(loss)
        me_optimizer.step()

        # reload state
        state = me_optimizer.state_dict()
        me_optimizer.load_state_dict(state)
        for k, v in me_optimizer.optimizer.state.items():
            self.assertTrue(k.dtype == torch.float16)
            for v_i in v.values():
                if torch.is_tensor(v_i):
                    self.assertTrue(v_i.dtype == torch.float32)


if __name__ == '__main__':
    unittest.main()
