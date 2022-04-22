# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
import logging
import torch.distributed as dist

class AdjustSmoothL1Loss(nn.Module):

    def __init__(self, num_features, momentum=0.1, beta=1. /9):
        super(AdjustSmoothL1Loss, self).__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.beta = beta
        self.register_buffer(
            'running_mean', torch.empty(num_features).fill_(beta)
        )
        self.register_buffer('running_var', torch.zeros(num_features))
        self.logger = logging.getLogger("maskrcnn_benchmark.trainer")

    def forward(self, inputs, target, size_average=True):

        n = torch.abs(inputs -target)
        n_mean = n.mean(dim=0)
        n_var = n.var(dim=0)
        with torch.no_grad():
            if torch.isnan(n_var).sum().item() == 0:
                self.running_mean = self.running_mean.to(n.device)
                self.running_mean *= (1 - self.momentum)
                self.running_mean += (self.momentum * n_mean)
                self.running_var = self.running_var.to(n.device)
                self.running_var *= (1 - self.momentum)
                self.running_var += (self.momentum * n_var)


        beta = (self.running_mean - self.running_var)
        beta = beta.clamp(max=self.beta, min=1e-3)

        beta = beta.view(-1, self.num_features).to(n.device)
        cond = n < beta.expand_as(n)
        loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
        if size_average:
            return loss.mean()
        return loss.sum()

