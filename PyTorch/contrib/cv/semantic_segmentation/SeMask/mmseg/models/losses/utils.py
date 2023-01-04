# -*- coding: utf-8 -*-
# BSD 3-Clause License
#
# Copyright (c) 2017
# All rights reserved.
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ==========================================================================

import functools

import torch.nn.functional as F


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        assert weight.dim() == loss.dim()
        if weight.dim() > 1:
            assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def weighted_loss(loss_func):
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    avg_factor=None, **kwargs)`.

    :Example:

    >>> import torch
    >>> @weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, avg_factor=2)
    tensor(1.5000)
    """

    @functools.wraps(loss_func)
    def wrapper(pred,
                target,
                weight=None,
                reduction='mean',
                avg_factor=None,
                **kwargs):
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    return wrapper
