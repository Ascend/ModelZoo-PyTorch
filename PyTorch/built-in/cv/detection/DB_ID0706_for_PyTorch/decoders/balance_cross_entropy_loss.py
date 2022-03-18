#     Copyright [yyyy] [name of copyright owner]
#     Copyright 2020 Huawei Technologies Co., Ltd
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
#

import torch
import torch.nn as nn

# for static shape
CONST_K = 4000000
CONST_K1 = 6000000
class BalanceCrossEntropyLoss(nn.Module):
    '''
    Balanced cross entropy loss.
    Shape:
        - Input: :math:`(N, 1, H, W)`
        - GT: :math:`(N, 1, H, W)`, same shape as the input
        - Mask: :math:`(N, H, W)`, same spatial shape as the input
        - Output: scalar.

    Examples::

        >>> m = nn.Sigmoid()
        >>> loss = nn.BCELoss()
        >>> input = torch.randn(3, requires_grad=True)
        >>> target = torch.empty(3).random_(2)
        >>> output = loss(m(input), target)
        >>> output.backward()
    '''

    def __init__(self, negative_ratio=3.0, eps=1e-6):
        super(BalanceCrossEntropyLoss, self).__init__()
        self.negative_ratio = negative_ratio
        self.eps = eps

    def forward(self,
                pred: torch.Tensor,
                gt: torch.Tensor,
                mask: torch.Tensor,
                return_origin=False):
        '''
        Args:
            pred: shape :math:`(N, 1, H, W)`, the prediction of network
            gt: shape :math:`(N, 1, H, W)`, the target
            mask: shape :math:`(N, H, W)`, the mask indicates positive regions
        '''
        positive = (gt * mask).float()
        negative = (mask - positive).float()
        positive_count = int(positive.sum())
        negative_count = min(int(negative.sum()),
                            int(positive_count * self.negative_ratio))
        loss = nn.functional.binary_cross_entropy(
            pred, gt, reduction='none')[:, 0, :, :]
        positive_loss = loss * positive
        negative_loss = (loss * negative).view(-1)
        global CONST_K
        global CONST_K1
        if negative_loss.shape[0] < CONST_K1:
            CONST_K = int(negative_loss.shape[0] * 0.4)
            CONST_K1 = int(negative_loss.shape[0] * 0.6)
        if negative_count <= CONST_K:
            negative_loss, _ = torch.topk(negative_loss.half(), CONST_K)
            negative_loss = negative_loss.float()[:negative_count]
            negative_loss = negative_loss.cpu()
        elif negative_count > CONST_K and negative_count <= CONST_K1:
            negative_loss, _ = torch.topk(negative_loss.half(), CONST_K1)
            negative_loss = negative_loss.float()[:negative_count]
            negative_loss = negative_loss.cpu()
        else:
            negative_loss = negative_loss.cpu()
            negative_loss, _ = torch.topk(negative_loss, negative_count)
        positive_sum = positive_loss.sum().cpu()
        balance_loss = (positive_sum + negative_loss.sum()) /\
            (positive_count + negative_count + self.eps)
        balance_loss = balance_loss.npu()
        if return_origin:
            return balance_loss, loss
        return balance_loss
