# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import numpy as np


class SmoothL1Loss(torch.nn.Module):
    def __init__(self, beta=1. / 9):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta

    def forward(self, input_, target, size_average=True):
        return smooth_l1_loss(input_, target, size_average=size_average)


def smooth_l1_loss(_input, target, alpha=0.5, gamma=1.5, beta=1.0, size_average=True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """

    diff = torch.abs(_input - target)
    b = np.e ** (gamma / alpha) - 1
    cond = diff < beta
    neg_cond = (~cond)
    loss = (alpha / b * (b * diff + 1) * torch.log(b * diff / beta + 1) - alpha * diff) * cond.half()
    loss = loss + (gamma * diff + gamma / b - alpha * beta) * neg_cond.half()
    if size_average:
        return loss.mean()
    return loss.sum()
