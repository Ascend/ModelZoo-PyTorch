# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += self.criterion(heatmap_pred.mul(target_weight[:, idx]),
                                       heatmap_gt.mul(target_weight[:, idx]))
            else:
                loss += self.criterion(heatmap_pred, heatmap_gt)

        return loss


class PerJointMSELoss(nn.Module):
    def __init__(self):
        super(PerJointMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='sum')

    def forward(self, output, target, use_target_weight = False, target_weight=None):
        if use_target_weight:
            batch_size = output.size(0)
            num_joints = output.size(1)

            heatmap_pred = output.reshape((batch_size, num_joints, -1))
            heatmap_gt = target.reshape((batch_size, num_joints, -1))
            loss = self.criterion(heatmap_pred.mul(target_weight), heatmap_gt.mul(target_weight))
        else:
            loss = self.criterion(output, target)

        return loss


class PerJointL1Loss(nn.Module):
    def __init__(self):
        super(PerJointL1Loss, self).__init__()
        self.criterion = nn.L1Loss(reduction='mean')

    def forward(self, output, target, use_target_weight=False, target_weight=None):
        if use_target_weight:
            batch_size = output.size(0)
            num_joints = output.size(1)

            pred = output.reshape((batch_size, num_joints, -1))
            gt = target.reshape((batch_size, num_joints, -1))
            loss = self.criterion(pred.mul(target_weight), gt.mul(target_weight))
        else:
            loss = self.criterion(output, target)

        return loss
