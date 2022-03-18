# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
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
#

import torch.nn as nn

from .registry import ACTIVATION_LAYERS


@ACTIVATION_LAYERS.register_module()
class HSigmoid(nn.Module):
    """Hard Sigmoid Module. Apply the hard sigmoid function:
    Hsigmoid(x) = min(max((x + bias) / divisor, min_value), max_value)
    Default: Hsigmoid(x) = min(max((x + 1) / 2, 0), 1)

    Args:
        bias (float): Bias of the input feature map. Default: 1.0.
        divisor (float): Divisor of the input feature map. Default: 2.0.
        min_value (float): Lower bound value. Default: 0.0.
        max_value (float): Upper bound value. Default: 1.0.

    Returns:
        Tensor: The output tensor.
    """

    def __init__(self, bias=1.0, divisor=2.0, min_value=0.0, max_value=1.0):
        super(HSigmoid, self).__init__()
        self.bias = bias
        self.divisor = divisor
        assert self.divisor != 0
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x):
        x = (x + self.bias) / self.divisor

        return x.clamp_(self.min_value, self.max_value)
