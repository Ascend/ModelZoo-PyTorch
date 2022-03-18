# Copyright (c) Soumith Chintala 2016,
# All rights reserved
#
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://spdx.org/licenses/BSD-3-Clause.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License
"""Custom losses."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

__all__ = ['ICNetLoss']

# TODO: optim function
class ICNetLoss(nn.CrossEntropyLoss):
    """Cross Entropy Loss for ICNet"""
    
    def __init__(self, aux_weight=0.4, ignore_index=-1):
        super(ICNetLoss, self).__init__(ignore_index=ignore_index)
        self.aux_weight = aux_weight

    def forward(self, *inputs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])

        pred, pred_sub4, pred_sub8, pred_sub16, target = tuple(inputs)
        # [batch, H, W] -> [batch, 1, H, W]
        target = target.unsqueeze(1).float()
        target_sub4 = F.interpolate(target, pred_sub4.size()[2:], mode='bilinear', align_corners=True).squeeze(1).int()
        target_sub8 = F.interpolate(target, pred_sub8.size()[2:], mode='bilinear', align_corners=True).squeeze(1).int()
        target_sub16 = F.interpolate(target, pred_sub16.size()[2:], mode='bilinear', align_corners=True).squeeze(1).int()       
        
        if 0:        
            loss1 = super(ICNetLoss, self).forward(pred_sub4, target_sub4)         
            loss2 = super(ICNetLoss, self).forward(pred_sub8, target_sub8)  
            loss3 = super(ICNetLoss, self).forward(pred_sub16, target_sub16)           
        else: # Temporarily circumvent
            loss1 = super(ICNetLoss, self).forward(pred_sub4.cpu(), target_sub4.to(torch.long).cpu()).npu()
            loss2 = super(ICNetLoss, self).forward(pred_sub8.cpu(), target_sub8.to(torch.long).cpu()).npu()
            loss3 = super(ICNetLoss, self).forward(pred_sub16.cpu(), target_sub16.to(torch.long).cpu()).npu()
        
        #return dict(loss=loss1 + loss2 * self.aux_weight + loss3 * self.aux_weight)
        return loss1 + loss2 * self.aux_weight + loss3 * self.aux_weight

