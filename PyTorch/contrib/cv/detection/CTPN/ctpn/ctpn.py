# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.contrib.npu.optimized_lib import module as nnn
from ctpn import config

"""
回归损失: smooth L1 Loss
只针对正样本求取回归损失
L = 0.5*x**2  |x|<1
L = |x| - 0.5
sigma: 平滑系数
1、从预测框p和真值框g中筛选出正样本
2、|x| = |g - p|
3、求取loss，这里设置了一个平滑系数 1/sigma
  (1) |x|>1/sigma: loss = |x| - 0.5/sigma
  (2) |x|<1/sigma: loss = 0.5*sigma*|x|**2
"""


class RPN_REGR_Loss(nn.Module):
    def __init__(self, device, sigma=9.0):
        super(RPN_REGR_Loss, self).__init__()
        self.sigma = sigma
        self.device = device

    def forward(self, input_data, target):
        input_data = input_data.to(self.device).float()
        target = target.to(self.device).float()
        cls = target[0, :, 0]
        regression = target[0, :, 1:3]
        regr_keep = (cls == 1).nonzero()[:, 0]
        regr_true = regression[regr_keep]
        if regr_true.numel() > 0:
            regr_pred = input_data[0][regr_keep]
            diff = torch.abs(regr_true - regr_pred)
            less_one = (diff < 1.0 / self.sigma).float()
            loss = less_one * 0.5 * diff ** 2 * self.sigma + torch.abs(1 - less_one) * (diff - 0.5 / self.sigma)
            loss = torch.sum(loss, 1)
            loss = torch.mean(loss)
        else:
            loss = input_data.sum() * 0
        return loss.to(self.device)


"""
分类损失: softmax loss
1、OHEM模式
  (1) 筛选出正样本，求取softmaxloss
  (2) 求取负样本数量N_neg, 指定样本数量N, 求取负样本的topK loss, 其中K = min(N_neg, N - len(pos_num))
  (3) loss = loss1 + loss2
2、求取NLLLoss，截断在(0, 10)区间
"""


class RPN_CLS_Loss(nn.Module):
    def __init__(self, device):
        super(RPN_CLS_Loss, self).__init__()
        self.device = device
        self.L_cls = nn.CrossEntropyLoss(reduction='none').to(self.device)

    def forward(self, input_data, target):
        input_data = input_data.to(self.device).float()
        target = target.to(self.device).float()
        if config.OHEM:
            cls_gt = target[0][0]
            num_pos = 0
            loss_pos_sum = 0

            if len((cls_gt == 1).nonzero()) != 0:
                cls_pos = (cls_gt == 1).nonzero()[:, 0]
                gt_pos = cls_gt[cls_pos].long()
                cls_pred_pos = input_data[0][cls_pos]
                loss_pos = self.L_cls(cls_pred_pos.view(-1, 2), gt_pos.view(-1))
                loss_pos_sum = loss_pos.sum()
                num_pos = len(loss_pos)

            cls_neg = (cls_gt == 0).nonzero()[:, 0]
            gt_neg = cls_gt[cls_neg].long()
            cls_pred_neg = input_data[0][cls_neg]

            loss_neg = self.L_cls(cls_pred_neg.view(-1, 2), gt_neg.view(-1))
            loss_neg_topK, _ = torch.topk(loss_neg, min(len(loss_neg), config.RPN_TOTAL_NUM - num_pos))
            loss_cls = loss_pos_sum + loss_neg_topK.sum()
            loss_cls = loss_cls / config.RPN_TOTAL_NUM

            return loss_cls.to(self.device)
        else:
            y_true = target[0][0]
            cls_keep = (y_true != -1).nonzero()[:, 0]
            cls_true = y_true[cls_keep].long()
            cls_pred = input_data[0][cls_keep]
            loss = F.nll_loss(F.log_softmax(cls_pred, dim=-1), cls_true)
            loss = torch.clamp(torch.mean(loss), 0, 10) if loss.numel() > 0 else torch.tensor(0.0)

            return loss.to(self.device)


class basic_conv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=True):
        super(basic_conv, self).__init__()
        self.out_channels = out_planes
        self.bn = bn
        self.relu = relu
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        if self.bn:
            self.batchnorm = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
        if self.relu:
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.batchnorm(x)
        if self.relu:
            x = self.act(x)
        return x


"""
image -> feature map -> rpn -> blstm -> fc -> classifier
                                           -> regression
"""


class CTPN_Model(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = models.resnet34(pretrained=True)
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        del base_model.maxpool
        del base_model.avgpool
        del base_model.fc
        self.rpn = basic_conv(512, 512, 3, 1, 1, bn=False)
        self.brnn = nnn.BiLSTM(512, 128)
        self.lstm_fc = basic_conv(256, 512, 1, 1, relu=True, bn=False)
        self.rpn_class = basic_conv(512, 10 * 2, 1, 1, relu=False, bn=False)
        self.rpn_regress = basic_conv(512, 10 * 2, 1, 1, relu=False, bn=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # rpn
        x = self.rpn(x)  # [b, c, h, w]

        x1 = x.permute(0, 2, 3, 1).contiguous()  # channels last   [b, h, w, c]
        b = x1.size()  # b, h, w, c
        x1 = x1.view(b[0] * b[1], b[2], b[3])

        x2 = self.brnn(x1)

        xsz = x.size()
        x3 = x2.view(xsz[0], xsz[2], xsz[3], 256)  # torch.Size([4, 20, 20, 256])

        x3 = x3.permute(0, 3, 1, 2).contiguous()  # channels first [b, c, h, w]
        x3 = self.lstm_fc(x3)
        x = x3

        cls = self.rpn_class(x)
        regression = self.rpn_regress(x)

        cls = cls.permute(0, 2, 3, 1).contiguous()
        regression = regression.permute(0, 2, 3, 1).contiguous()

        cls = cls.view(cls.size(0), cls.size(1) * cls.size(2) * 10, 2)
        regression = regression.view(regression.size(0), regression.size(1) * regression.size(2) * 10, 2)

        return cls, regression


if __name__ == '__main__':
    CTPN_Model()
