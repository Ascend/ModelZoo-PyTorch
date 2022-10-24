# Copyright 2021 Huawei Technologies Co., Ltd
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
import torch.nn as nn

from ..builder import BACKBONES
from .base_backbone import BaseBackbone


@BACKBONES.register_module()
class LeNet5(BaseBackbone):
    """`LeNet5 <https://en.wikipedia.org/wiki/LeNet>`_ backbone.

    The input for LeNet-5 is a 32×32 grayscale image.

    Args:
        num_classes (int): number of classes for classification.
            The default value is -1, which uses the backbone as
            a feature extractor without the top classifier.
    """

    def __init__(self, num_classes=-1):
        super(LeNet5, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1), nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1), nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(16, 120, kernel_size=5, stride=1), nn.Tanh())
        if self.num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Linear(120, 84),
                nn.Tanh(),
                nn.Linear(84, num_classes),
            )

    def forward(self, x):

        x = self.features(x)
        if self.num_classes > 0:
            x = self.classifier(x.squeeze())

        return (x, )
