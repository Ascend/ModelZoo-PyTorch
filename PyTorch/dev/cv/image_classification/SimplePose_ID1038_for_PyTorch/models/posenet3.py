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
Still in development.
Light 4-stage IMHN, change the layers_transposed.py:
        self.resBlock = resBlock
        self.convBlock = convBlock
"""
import math
import torch
from torch import nn
from models.layers_transposed import Conv, Hourglass, SELayer, Backbone
from models.loss_model_parallel import MultiTaskLossParallel
from models.loss_model import MultiTaskLoss
from torchvision.models import densenet


class Merge(nn.Module):
    """Change the channel dimension of the input tensor"""

    def __init__(self, x_dim, y_dim, bn=False):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=bn)

    def forward(self, x):
        return self.conv(x)


class Features(nn.Module):
    """Input: feature maps produced by hourglass block
       Return: 5 different scales of feature maps, 128*128, 64*64, 32*32, 16*16, 8*8"""

    def __init__(self, inp_dim, increase=128, bn=False):
        super(Features, self).__init__()
        # Regress 5 different scales of heatmaps per stack
        self.before_regress = nn.ModuleList(
            [nn.Sequential(
                           Conv(inp_dim + i * increase, inp_dim + i * increase, 3, bn=bn, dropout=False),
                           ) for i in range(5)])

    def forward(self, fms):
        assert len(fms) == 5, "hourglass output {} tensors,but 5 scale heatmaps are supervised".format(len(fms))
        return [self.before_regress[i](fms[i]) for i in range(5)]


class PoseNet(nn.Module):
    def __init__(self, nstack, inp_dim, oup_dim, bn=False, increase=128, init_weights=True, **kwargs):
        """
        Pack or initialize the trainable parameters of the network
        :param nstack: number of stack
        :param inp_dim: input tensor channels fed into the hourglass block
        :param oup_dim: channels of regressed feature maps
        :param bn: use batch normalization
        :param increase: increased channels once down-sampling
        :param kwargs:
        """
        super(PoseNet, self).__init__()
        self.pre = nn.Sequential(
            Conv(3, 64, 7, 2, bn=bn),
            Conv(64, 128, bn=bn),
            nn.MaxPool2d(2, 2),
            Conv(128, 128, bn=bn),
            Conv(128, inp_dim, bn=bn)
        )
        # self.pre = Backbone(nFeat=inp_dim)  # It doesn't affect the results regardless of which self.pre is used
        self.hourglass = nn.ModuleList([Hourglass(4, inp_dim, increase, bn=bn) for _ in range(nstack)])
        self.features = nn.ModuleList([Features(inp_dim, increase=increase, bn=bn) for _ in range(nstack)])
        # predict 5 different scales of heatmpas per stack, keep in mind to pack the list using ModuleList.
        # Notice: nn.ModuleList can only identify Module subclass! Thus, we must pack the inner layers in ModuleList.
        # TODO: change the outs layers, Conv(inp_dim, oup_dim, 1, relu=False, bn=False)
        self.outs = nn.ModuleList(
            [nn.ModuleList([Conv(inp_dim + j * increase, oup_dim, 1, relu=False, bn=False) for j in range(5)]) for i in
             range(nstack)])
        self.channel_attention = nn.ModuleList(
            [nn.ModuleList([SELayer(inp_dim + j * increase) for j in range(5)]) for i in
             range(nstack)])

        # TODO: change the merge layers, Merge(inp_dim, inp_dim + j * increase)
        self.merge_features = nn.ModuleList(
            [nn.ModuleList([Merge(inp_dim + j * increase, inp_dim + j * increase, bn=bn) for j in range(5)]) for i in
             range(nstack - 1)])
        self.merge_preds = nn.ModuleList(
            [nn.ModuleList([Merge(oup_dim, inp_dim + j * increase, bn=bn) for j in range(5)]) for i in
             range(nstack - 1)])
        self.nstack = nstack
        if init_weights:
            self._initialize_weights()

    def forward(self, imgs):
        # Input Tensor: a batch of images within [0,1], shape=(N, H, W, C). Pre-processing was done in data generator
        x = imgs.permute(0, 3, 1, 2)  # Permute the dimensions of images to (N, C, H, W)
        x = self.pre(x)
        pred = []
        # loop over stack
        for i in range(self.nstack):
            preds_instack = []
            # return 5 scales of feature maps
            hourglass_feature = self.hourglass[i](x)

            if i == 0:  # cache for smaller feature maps produced by hourglass block
                features_cache = [torch.zeros_like(hourglass_feature[scale]) for scale in range(5)]
                for s in range(5):  # channel attention before heatmap regression
                    hourglass_feature[s] = self.channel_attention[i][s](hourglass_feature[s])
            else:  # residual connection across stacks
                for k in range(5):
                    #  python閲岄潰鐨+=, 锛*=涔熸槸in-place operation,闇瑕佹敞鎰
                    hourglass_feature_attention = self.channel_attention[i][k](hourglass_feature[k])

                    hourglass_feature[k] = hourglass_feature_attention + features_cache[k]
            # feature maps before heatmap regression
            features_instack = self.features[i](hourglass_feature)

            for j in range(5):  # handle 5 scales of heatmaps
                preds_instack.append(self.outs[i][j](features_instack[j]))
                if i != self.nstack - 1:
                    if j == 0:
                        x = x + self.merge_preds[i][j](preds_instack[j]) + self.merge_features[i][j](
                            features_instack[j])  # input tensor for next stack
                        features_cache[j] = self.merge_preds[i][j](preds_instack[j]) + self.merge_features[i][j](
                            features_instack[j])

                    else:
                        # reset the res caches
                        features_cache[j] = self.merge_preds[i][j](preds_instack[j]) + self.merge_features[i][j](
                            features_instack[j])
            pred.append(preds_instack)
        # returned list shape: [nstack * [batch*128*128, batch*64*64, batch*32*32, batch*16*16, batch*8*8]]z
        return pred

    def _initialize_weights(self):
        for m in self.modules():
            # 鍗风Н鐨勫垵濮嬪寲鏂规硶
            if isinstance(m, nn.Conv2d):
                # TODO: 浣跨敤姝ｆ佸垎甯冭繘琛屽垵濮嬪寲锛0, 0.01) 缃戠粶鏉冮噸鐪嬬湅
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # He kaiming 鍒濆鍖, 鏂瑰樊涓2/n. math.sqrt(2. / n) 鎴栬呯洿鎺ヤ娇鐢ㄧ幇鎴愮殑nn.init涓殑鍑芥暟銆傚湪杩欓噷浼氭搴︾垎鐐
                m.weight.data.normal_(0, 0.001)    # # math.sqrt(2. / n)
                # torch.nn.init.kaiming_normal_(m.weight)
                # bias閮藉垵濮嬪寲涓0
                if m.bias is not None:  # 褰撴湁BN灞傛椂锛屽嵎绉眰Con涓嶅姞bias锛
                    m.bias.data.zero_()
            # batchnorm浣跨敤鍏1鍒濆鍖 bias鍏0
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)  # todo: 0.001?
                # m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class Network(torch.nn.Module):
    """
    Wrap the network module as well as the loss module on all GPUs to balance the computation among GPUs.
    """
    def __init__(self, opt, config, bn=False, dist=False, swa=False):
        super(Network, self).__init__()
        self.posenet = PoseNet(opt.nstack, opt.hourglass_inp_dim, config.num_layers, bn=bn, increase=opt.increase)
        # If we use train_parallel, we implement the parallel loss . And if we use train_distributed,
        # we should use single process loss because each process on these 4 GPUs  is independent
        self.criterion = MultiTaskLoss(opt, config) if dist else MultiTaskLossParallel(opt, config)
        self.swa = swa

    def forward(self, input_all):
        # Batch will be divided and Parallel Model will call this forward on every GPU
        inp_imgs = input_all[0]
        target_tuple = input_all[1:]
        output_tuple = self.posenet(inp_imgs)

        if not self.training:  # testing mode
            loss = self.criterion(output_tuple, target_tuple)
            # output will be concatenated  along batch channel automatically after the parallel model return
            return output_tuple, loss

        else:  # training mode
            if not self.swa:
                loss = self.criterion(output_tuple, target_tuple)

                # output will be concatenated  along batch channel automatically after the parallel model return
                return loss
            else:
                return output_tuple


class NetworkEval(torch.nn.Module):
    """
    Wrap the network module as well as the loss module on all GPUs to balance the computation among GPUs.
    """
    def __init__(self, opt, config, bn=False):
        super(NetworkEval, self).__init__()
        self.posenet = PoseNet(opt.nstack, opt.hourglass_inp_dim, config.num_layers, bn=bn, init_weights=False,
                               increase=opt.increase)

    def forward(self, inp_imgs):
        # Batch will be divided and Parallel Model will call this forward on every GPU
        output_tuple = self.posenet(inp_imgs)

        if not self.training:
            # output will be concatenated  along batch channel automatically after the parallel model return
            return output_tuple
        else:
            # output will be concatenated  along batch channel automatically after the parallel model return
            raise ValueError('\nOnly eval mode is available!!')


if __name__ == '__main__':
    from time import time

    pose = PoseNet(4, 256, 54, bn=True)  # .npu()
    for param in pose.parameters():
        if param.requires_grad:
            print('param autograd')
            break

    t0 = time()
    input = torch.rand(1, 128, 128, 3)  # .npu()
    print(pose)
    output = pose(input)  # type: torch.Tensor

    output[0][0].sum().backward()

    t1 = time()
    print('********** Consuming Time is: {} second  **********'.format(t1 - t0))

    # #
    # import torch.onnx
    #
    # pose = PoseNet(4, 256, 34)
    # dummy_input = torch.randn(1, 512, 512, 3)
    # torch.onnx.export(pose, dummy_input, "posenet.onnx")  # netron --host=localhost
