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
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from ..builder import NECKS


def gem(x: Tensor, p: Parameter, eps: float = 1e-6, clamp=True) -> Tensor:
    if clamp:
        x = x.clamp(min=eps)
    return F.avg_pool2d(x.pow(p), (x.size(-2), x.size(-1))).pow(1. / p)


@NECKS.register_module()
class GeneralizedMeanPooling(nn.Module):
    """Generalized Mean Pooling neck.

    Note that we use `view` to remove extra channel after pooling. We do not
    use `squeeze` as it will also remove the batch dimension when the tensor
    has a batch dimension of size 1, which can lead to unexpected errors.

    Args:
        p (float): Parameter value.
            Default: 3.
        eps (float): epsilon.
            Default: 1e-6
        clamp (bool): Use clamp before pooling.
            Default: True
    """

    def __init__(self, p=3., eps=1e-6, clamp=True):
        assert p >= 1, "'p' must be a value greater then 1"
        super(GeneralizedMeanPooling, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps
        self.clamp = clamp

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            outs = tuple([
                gem(x, p=self.p, eps=self.eps, clamp=self.clamp)
                for x in inputs
            ])
            outs = tuple(
                [out.view(x.size(0), -1) for out, x in zip(outs, inputs)])
        elif isinstance(inputs, torch.Tensor):
            outs = gem(inputs, p=self.p, eps=self.eps, clamp=self.clamp)
            outs = outs.view(inputs.size(0), -1)
        else:
            raise TypeError('neck inputs should be tuple or torch.tensor')
        return outs
