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
import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub

class RepVGGWholeQuant(nn.Module):

    def __init__(self, repvgg_model, quantlayers):
        super(RepVGGWholeQuant, self).__init__()
        assert quantlayers in ['all', 'exclud_first_and_linear', 'exclud_first_and_last']
        self.quantlayers = quantlayers
        self.quant = QuantStub()
        self.stage0, self.stage1, self.stage2, self.stage3, self.stage4 = repvgg_model.stage0, repvgg_model.stage1, repvgg_model.stage2, repvgg_model.stage3, repvgg_model.stage4
        self.gap, self.linear = repvgg_model.gap, repvgg_model.linear
        self.dequant = DeQuantStub()


    def forward(self, x):
        if self.quantlayers == 'all':
            x = self.quant(x)
            out = self.stage0(x)
        else:
            out = self.stage0(x)
            out = self.quant(out)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        if self.quantlayers == 'all':
            out = self.stage4(out)
            out = self.gap(out).view(out.size(0), -1)
            out = self.linear(out)
            out = self.dequant(out)
        elif self.quantlayers == 'exclud_first_and_linear':
            out = self.stage4(out)
            out = self.dequant(out)
            out = self.gap(out).view(out.size(0), -1)
            out = self.linear(out)
        else:
            out = self.dequant(out)
            out = self.stage4(out)
            out = self.gap(out).view(out.size(0), -1)
            out = self.linear(out)
        return out

    #   From https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html
    def fuse_model(self):
        for m in self.modules():
            if type(m) == nn.Sequential and hasattr(m, 'conv'):
                # Note that we moved ReLU from "block.nonlinearity" into "rbr_reparam" (nn.Sequential).
                # This makes it more convenient to fuse operators using off-the-shelf APIs.
                torch.quantization.fuse_modules(m, ['conv', 'bn', 'relu'], inplace=True)

    def _get_qconfig(self):
        return torch.quantization.get_default_qat_qconfig('fbgemm')

    def prepare_quant(self):
        #   From https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html
        self.fuse_model()
        qconfig = self._get_qconfig()
        self.qconfig = qconfig
        torch.quantization.prepare_qat(self, inplace=True)

    def freeze_quant_bn(self):
        self.apply(torch.nn.intrinsic.qat.freeze_bn_stats)