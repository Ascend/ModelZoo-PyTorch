#     Copyright 2021 Huawei Technologies Co., Ltd
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

"""Library implementing activation functions.

Authors
 * Mirco Ravanelli 2020
 * Jianyuan Zhong 2020
"""

import torch
import logging
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class Softmax(torch.nn.Module):
    """Computes the softmax of a 2d, 3d, or 4d input tensor.

    Arguments
    ---------
    apply_log : bool
        Whether to apply the log function before softmax.
    dim : int
        If the dimension where softmax is applied.

    Example
    -------
    >>> classifier = Softmax()
    >>> inputs = torch.rand(10, 50, 40)
    >>> output = classifier(inputs)
    >>> output.shape
    torch.Size([10, 50, 40])
    """

    def __init__(self, apply_log=False, dim=-1):
        super().__init__()

        if apply_log:
            self.act = torch.nn.LogSoftmax(dim=dim)
        else:
            self.act = torch.nn.Softmax(dim=dim)

    def forward(self, x):
        """Returns the softmax of the input tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.
        """
        # Reshaping the tensors
        dims = x.shape

        if len(dims) == 3:
            x = x.reshape(dims[0] * dims[1], dims[2])

        if len(dims) == 4:
            x = x.reshape(dims[0] * dims[1], dims[2], dims[3])

        x_act = self.act(x)

        # Retrieving the original shape format
        if len(dims) == 3:
            x_act = x_act.reshape(dims[0], dims[1], dims[2])

        if len(dims) == 4:
            x_act = x_act.reshape(dims[0], dims[1], dims[2], dims[3])

        return x_act


class GumbelSoftmax(torch.nn.Module):
    """Samples from the Gumbel-Softmax distribution and optionally discretizes.

    Reference: https://arxiv.org/abs/1611.00712, https://arxiv.org/abs/1611.01144

    Arguments
    ----------
    tau: float
        non-negative scalar temperature
    hard: bool
        if True, the returned samples will be discretized as one-hot vectors, but will be differentiated as if it is the soft sample in autograd
    dim: int
        A dimension along which softmax will be computed (default: -1).

    Example
    -------
    >>> x = torch.randn((8, 40, 120))
    >>> act = GumbelSoftmax(0.8, True)
    >>> x = act(x)
    """

    def __init__(self, tau, hard=False, apply_log=False):
        super().__init__()
        self.tau = tau
        self.hard = hard
        self.apply_log = apply_log

    def forward(self, x):
        with torch.autograd.profiler.record_function("gumbel_softmax"):
            ret_gum = F.gumbel_softmax(x, tau=self.tau, hard=self.hard)
        if self.apply_log:
            return torch.log(ret_gum)
        return ret_gum


class Swish(torch.nn.Module):
    """ The class implements the Swish activation function from
    https://arxiv.org/pdf/2005.03191.pdf

    given input x. Swish(x) = x / (1 + exp(beta * x))

    Arguments
    ---------
    beta: float
        Beta value.

    Example
    -------
    >>> x = torch.randn((8, 40, 120))
    >>> act = Swish()
    >>> x = act(x)
    """

    def __init__(self, beta=1):
        super().__init__()
        self.beta = beta
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        """Returns the Swished input tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.
        """
        return x * self.sigmoid(self.beta * x)
