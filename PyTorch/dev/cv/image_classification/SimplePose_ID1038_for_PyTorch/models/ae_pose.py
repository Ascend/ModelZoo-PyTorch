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
"""
Copied from Associative Embedding pytorch project. Hourglass Network
"""
import torch
from torch.autograd import Variable
from torch import nn
from models.ae_layer import Conv, Hourglass
from models.loss_model_parallel import MultiTaskLossParallel
from models.loss_model import MultiTaskLoss


class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)


class PoseNet(nn.Module):
    def __init__(self, nstack, inp_dim, oup_dim, bn=False, increase=128, init_weights=True, **kwargs):
        super(PoseNet, self).__init__()
        self.pre = nn.Sequential(
            Conv(3, 64, 7, 2, bn=bn),
            Conv(64, 128, bn=bn),
            nn.MaxPool2d(2, 2),
            Conv(128, 128, bn=bn),
            Conv(128, inp_dim, bn=bn)
        )
        self.features = nn.ModuleList([
            nn.Sequential(
                Hourglass(4, inp_dim, bn, increase),
                Conv(inp_dim, inp_dim, 3, bn=False),
                Conv(inp_dim, inp_dim, 3, bn=False)
            ) for i in range(nstack)])  # 鏋勯犱簡nstack涓繖鏍风殑Hourglass+2*Conve妯″潡

        self.outs = nn.ModuleList([Conv(inp_dim, oup_dim, 1, relu=False, bn=False) for i in range(nstack)])
        self.merge_features = nn.ModuleList([Merge(inp_dim, inp_dim) for i in range(nstack - 1)])
        self.merge_preds = nn.ModuleList([Merge(oup_dim, inp_dim) for i in range(nstack - 1)])

        self.nstack = nstack
        if init_weights:
            self._initialize_weights()

    def forward(self, imgs):
        x = imgs.permute(0, 3, 1, 2)
        x = self.pre(x)
        preds = []
        for i in range(self.nstack):
            preds_instack = []
            feature = self.features[i](x)
            preds_instack.append(self.outs[i](feature))
            if i != self.nstack - 1:
                x = x + self.merge_preds[i](preds_instack[-1]) + self.merge_features[i](feature)
            preds.append(preds_instack)
        return preds

    def _initialize_weights(self):
        for m in self.modules():
            # 鍗风Н鐨勫垵濮嬪寲鏂规硶
            if isinstance(m, nn.Conv2d):
                # TODO: 浣跨敤姝ｆ佸垎甯冭繘琛屽垵濮嬪寲锛0, 0.01) 缃戠粶鏉冮噸鐪嬬湅
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # He kaiming 鍒濆鍖, 鏂瑰樊涓2/n. math.sqrt(2. / n) 鎴栬呯洿鎺ヤ娇鐢ㄧ幇鎴愮殑nn.init涓殑鍑芥暟銆傚湪杩欓噷浼氭搴︾垎鐐
                m.weight.data.normal_(0, 0.01)    # # math.sqrt(2. / n)
                # torch.nn.init.uniform_(tensorx)
                # bias閮藉垵濮嬪寲涓0
                if m.bias is not None:  # 褰撴湁BN灞傛椂锛屽嵎绉眰Con涓嶅姞bias锛
                    m.bias.data.zero_()
            # batchnorm浣跨敤鍏1鍒濆鍖 bias鍏0
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)  # m.weight.data.normal_(0, 0.01) m.bias.data.zero_()


class Network(torch.nn.Module):
    """
    Wrap the network module as well as the loss module on all GPUs to balance the computation among GPUs.
    """
    def __init__(self, opt, config, bn=False, dist=False):
        super(Network, self).__init__()
        self.posenet = PoseNet(opt.nstack, opt.hourglass_inp_dim, config.num_layers, bn=bn)
        # If we use train_parallel, we implement the parallel loss . And if we use train_distributed,
        # we should use single process loss because each process on these 4 GPUs  is independent
        self.criterion = MultiTaskLoss(opt, config) if dist else MultiTaskLossParallel(opt, config)

    def forward(self, inp_imgs, target_tuple):
        # Batch will be divided and Parallel Model will call this forward on every GPU
        output_tuple = self.posenet(inp_imgs)
        loss = self.criterion(output_tuple, target_tuple)

        if not self.training:
            # output will be concatenated  along batch channel automatically after the parallel model return
            return output_tuple, loss
        else:
            # output will be concatenated  along batch channel automatically after the parallel model return
            return loss


class NetworkEval(torch.nn.Module):
    """
    Wrap the network module as well as the loss module on all GPUs to balance the computation among GPUs.
    """
    def __init__(self, opt, config, bn=False):
        super(NetworkEval, self).__init__()
        self.posenet = PoseNet(opt.nstack, opt.hourglass_inp_dim, config.num_layers, bn=bn)

    def forward(self, inp_imgs):
        # Batch will be divided and Parallel Model will call this forward on every GPU
        output_tuple = self.posenet(inp_imgs)

        if not self.training:
            # output will be concatenated  along batch channel automatically after the parallel model return
            return output_tuple
        else:
            # output will be concatenated  along batch channel automatically after the parallel model return
            raise ValueError('\nOnly eval mode is available!!')
