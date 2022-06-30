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
""" Test Time Pooling (Average-Max Pool)

Hacked together by / Copyright 2020 Ross Wightman
"""

import logging
from torch import nn
import torch.nn.functional as F

from .adaptive_avgmax_pool import adaptive_avgmax_pool2d
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')


_logger = logging.getLogger(__name__)


class TestTimePoolHead(nn.Module):
    def __init__(self, base, original_pool=7):
        super(TestTimePoolHead, self).__init__()
        self.base = base
        self.original_pool = original_pool
        base_fc = self.base.get_classifier()
        if isinstance(base_fc, nn.Conv2d):
            self.fc = base_fc
        else:
            self.fc = nn.Conv2d(
                self.base.num_features, self.base.num_classes, kernel_size=1, bias=True)
            self.fc.weight.data.copy_(base_fc.weight.data.view(self.fc.weight.size()))
            self.fc.bias.data.copy_(base_fc.bias.data.view(self.fc.bias.size()))
        self.base.reset_classifier(0)  # delete original fc layer

    def forward(self, x):
        x = self.base.forward_features(x)
        x = F.avg_pool2d(x, kernel_size=self.original_pool, stride=1)
        x = self.fc(x)
        x = adaptive_avgmax_pool2d(x, 1)
        return x.view(x.size(0), -1)


def apply_test_time_pool(model, config, use_test_size=True):
    test_time_pool = False
    if not hasattr(model, 'default_cfg') or not model.default_cfg:
        return model, False
    if use_test_size and 'test_input_size' in model.default_cfg:
        df_input_size = model.default_cfg['test_input_size']
    else:
        df_input_size = model.default_cfg['input_size']
    if config['input_size'][-1] > df_input_size[-1] and config['input_size'][-2] > df_input_size[-2]:
        _logger.info('Target input size %s > pretrained default %s, using test time pooling' %
                     (str(config['input_size'][-2:]), str(df_input_size[-2:])))
        model = TestTimePoolHead(model, original_pool=model.default_cfg['pool_size'])
        test_time_pool = True
    return model, test_time_pool
