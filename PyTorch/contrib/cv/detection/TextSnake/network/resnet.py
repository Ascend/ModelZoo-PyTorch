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
import torchvision.models.resnet as resnet
import torch.nn as nn

class ResNet50(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = resnet.resnet50(pretrained=True)

        self.stage1 = nn.Sequential(
            self.net.conv1,
            self.net.bn1,
            self.net.relu,
            self.net.maxpool
        )
        self.stage2 = self.net.layer1
        self.stage3 = self.net.layer2
        self.stage4 = self.net.layer3
        self.stage5 = self.net.layer4

    def forward(self, x):
        C1 = self.stage1(x)
        C2 = self.stage2(C1)
        C3 = self.stage3(C2)
        C4 = self.stage4(C3)
        C5 = self.stage5(C4)
        return C1, C2, C3, C4, C5


if __name__ == '__main__':
    import torch
    input = torch.randn((4, 3, 512, 512))
    net = ResNet50()
    C1, C2, C3, C4, C5 = net(input)
    print(C1.size())
    print(C2.size())
    print(C3.size())
    print(C4.size())
    print(C5.size())
