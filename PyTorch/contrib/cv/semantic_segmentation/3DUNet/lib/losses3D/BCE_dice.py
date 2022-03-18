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


import torch.nn as nn
from lib.losses3D.dice import DiceLoss
from lib.losses3D.basic import expand_as_one_hot
# Code was adapted and mofified from https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/unet3d/losses.py


class BCEDiceLoss(nn.Module):
    """Linear combination of BCE and Dice losses3D"""

    def __init__(self, alpha=1, beta=1, classes=4):
        super(BCEDiceLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.beta = beta
        self.dice = DiceLoss(classes=classes)
        self.classes=classes

    def forward(self, input, target):
        target_expanded = expand_as_one_hot(target.long(), self.classes)
        assert input.size() == target_expanded.size(), "'input' and 'target' must have the same shape"
        loss_1 = self.alpha * self.bce(input, target_expanded)
        loss_2, channel_score = self.beta * self.dice(input, target_expanded)
        return  (loss_1+loss_2) , channel_score
