# Copyright 2020 Huawei Technologies Co., Ltd
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

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import resnet


class WideBottleneck(resnet.Bottleneck):
    expansion = 2


def generate_model(model_depth, k, **kwargs):
    assert model_depth in [50, 101, 152, 200]

    inplanes = [x * k for x in resnet.get_inplanes()]
    if model_depth == 50:
        model = resnet.ResNet(WideBottleneck, [3, 4, 6, 3], inplanes, **kwargs)
    elif model_depth == 101:
        model = resnet.ResNet(WideBottleneck, [3, 4, 23, 3], inplanes, **kwargs)
    elif model_depth == 152:
        model = resnet.ResNet(WideBottleneck, [3, 8, 36, 3], inplanes, **kwargs)
    elif model_depth == 200:
        model = resnet.ResNet(WideBottleneck, [3, 24, 36, 3], inplanes,
                              **kwargs)

    return model
