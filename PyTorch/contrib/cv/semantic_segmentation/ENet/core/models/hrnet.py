# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""High-Resolution Representations for Semantic Segmentation"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class HRNet(nn.Module):
    """HRNet

        Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    aux : bool
        Auxiliary loss.
    Reference:
        Ke Sun. "High-Resolution Representations for Labeling Pixels and Regions."
        arXiv preprint arXiv:1904.04514 (2019).
    """
    def __init__(self, nclass, backbone='', aux=False, pretrained_base=False, **kwargs):
        super(HRNet, self).__init__()

    def forward(self, x):
        pass