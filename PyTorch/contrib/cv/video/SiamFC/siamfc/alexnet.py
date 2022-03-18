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

import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from .custom_transforms import ToTensor

from torchvision.models import alexnet
from torch.autograd import Variable
import torch.nn as nn

from .config import config


class SiameseAlexNet(nn.Module):
    def __init__(self):
        super(SiameseAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),  #
            nn.MaxPool2d(3, 2),
            nn.Conv2d(96, 256, 5, 1, groups=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, 3, 1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, 1, groups=2),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, 1, groups=2)
        )
        self.corr_bias = nn.Parameter(torch.zeros(1))  # Parameter
        self.exemplar = None

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        exemplar, instance = x  # x = ( exemplar, instance )
        # Train
        if exemplar is not None and instance is not None:  #
            # batch_size = exemplar.shape[0]
            exemplar = self.features(exemplar)  # batch, 256, 6, 6
            instance = self.features(instance)  # batch, 256, 20, 20
            N, C, H, W = instance.shape
            instance = instance.view(1, -1, H, W)
            score = F.conv2d(instance, exemplar, groups=N) * config.response_scale + self.corr_bias
            return score.transpose(0, 1)
        # Test(first frame)
        elif exemplar is not None and instance is None:
            self.exemplar = self.features(exemplar)  # 1, 256, 6, 6
            self.exemplar = torch.cat([self.exemplar for _ in range(3)], dim=0)  # 3, 256, 6, 6
        # Test(not first frame)
        else:
            # inference used we don't need to scale the response or add bias
            instance = self.features(instance)  # 3 scale
            N, _, H, W = instance.shape
            instance = instance.view(1, -1, H, W)  # 1, NxC, H, W
            score = F.conv2d(instance, self.exemplar, groups=N)
            return score.transpose(0, 1)


# generate label and weight
def _create_gt_mask(shape):
    # same for all pairs
    h, w = shape                                     # shape=[15,15]  255x255 - 17x17   (255-2*8)x(255-2*8) - 15x15
    y = np.arange(h, dtype=np.float32) - (h-1) / 2.  # [0,1,2...,14]-(15-1)/2-->y=[-7, -6 ,...0,...,6,7]
    x = np.arange(w, dtype=np.float32) - (w-1) / 2.  # [0,1,2...,14]-(15-1)/2-->x=[-7, -6 ,...0,...,6,7]
    y, x = np.meshgrid(y, x)
    dist = np.sqrt(x**2 + y**2)                      # ||u-c|| distance from the center point
    mask = np.zeros((h, w))
    mask[dist <= config.radius / config.total_stride] = 1  # ||u-c||*total_stride <= radius
    mask = mask[np.newaxis, :, :]                          # mask.shape=(1,15,15)
    weights = np.ones_like(mask)
    weights[mask == 1] = 0.5 / np.sum(mask == 1)   # 0.5/num(positive sample)
    weights[mask == 0] = 0.5 / np.sum(mask == 0)   # 0.5/num(negative sample)
    mask = np.repeat(mask, config.train_batch_size, axis=0)[:, np.newaxis, :, :]
    return mask.astype(np.float32), weights.astype(np.float32)  # mask.shape=(8,1,15,15)


# loss function
class Criterion(nn.Module):
    def __init__(self):
        super(Criterion, self).__init__()

    def forward(self, pred, label, weight):
        return F.binary_cross_entropy_with_logits(pred, label, weight,
                                                  reduction='sum') / config.train_batch_size  # normalize the batch_size
