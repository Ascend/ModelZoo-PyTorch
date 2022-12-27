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
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, Dropout, Sequential, Module
from helpers import get_blocks, Flatten, BottleneckIr, BottleneckIRSE, l2_norm


class Backbone(Module):
    def __init__(self, input_size, num_layers, mode='ir', drop_ratio=0.4, affine=True):
        super(Backbone, self).__init__()
        assert input_size in [112, 224], "input_size should be 112 or 224"
        assert num_layers in [50, 100, 152], "num_layers should be 50, 100 or 152"
        assert mode in ['ir', 'ir_se'], "mode should be ir or ir_se"
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = BottleneckIr
        elif mode == 'ir_se':
            unit_module = BottleneckIRSE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        if input_size == 112:
            self.output_layer = Sequential(BatchNorm2d(512),
                                           Dropout(drop_ratio),
                                           Flatten(),
                                           Linear(512 * 7 * 7, 512),
                                           BatchNorm1d(512, affine=affine))
        else:
            self.output_layer = Sequential(BatchNorm2d(512),
                                           Dropout(drop_ratio),
                                           Flatten(),
                                           Linear(512 * 14 * 14, 512),
                                           BatchNorm1d(512, affine=affine))

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return l2_norm(x)


def ir_50(input_size):
    """Constructs a ir-50 model."""
    model = Backbone(input_size, num_layers=50, mode='ir', drop_ratio=0.4, affine=False)
    return model


def ir_101(input_size):
    """Constructs a ir-101 model."""
    model = Backbone(input_size, num_layers=100, mode='ir', drop_ratio=0.4, affine=False)
    return model


def ir_152(input_size):
    """Constructs a ir-152 model."""
    model = Backbone(input_size, num_layers=152, mode='ir', drop_ratio=0.4, affine=False)
    return model


def ir_se_50(input_size):
    """Constructs a ir_se-50 model."""
    model = Backbone(input_size, num_layers=50, mode='ir_se', drop_ratio=0.4, affine=False)
    return model


def ir_se_101(input_size):
    """Constructs a ir_se-101 model."""
    model = Backbone(input_size, num_layers=100, mode='ir_se', drop_ratio=0.4, affine=False)
    return model


def ir_se_152(input_size):
    """Constructs a ir_se-152 model."""
    model = Backbone(input_size, num_layers=152, mode='ir_se', drop_ratio=0.4, affine=False)
    return model
