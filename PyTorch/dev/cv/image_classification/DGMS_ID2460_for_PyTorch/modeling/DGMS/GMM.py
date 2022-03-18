#
# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
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
# ============================================================================
#
# Copyright (c) Runpei Dong, ArChip Lab.

""" DGMS GM Sub-distribution implementation.

Author: Runpei Dong
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import config as cfg

from utils.misc import cluster_weights
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')

class GaussianMixtureModel(nn.Module):
    """Concrete GMM for sub-distribution approximation.
    """
    def __init__(self, num_components, init_weights, temperature=0.01, init_method="k-means"):
        super(GaussianMixtureModel, self).__init__()
        self.num_components = num_components
        self.temperature = temperature
        if torch.npu.is_available():
            self.device = torch.device(f'npu:{NPU_CALCULATE_DEVICE}')
        else:
            self.device = torch.device(f'npu:{NPU_CALCULATE_DEVICE}')
        self.params_initialization(init_weights, init_method)

    def params_initialization(self, init_weights, method='k-means'):
        """ Initialization of GMM parameters using k-means algorithm. """
        self.mu_zero = torch.tensor([0.0], device=self.device).float()
        self.pi_k, self.mu, self.sigma = \
                torch.ones(self.num_components-1, device=self.device), \
                torch.ones(self.num_components-1, device=self.device), \
                torch.ones(self.num_components-1, device=self.device)
        if method == 'k-means':
            initial_region_saliency, pi_init, pi_zero_init, sigma_init, _sigma_zero = cluster_weights(init_weights, self.num_components)
        elif method == 'empirical':
            initial_region_saliency, pi_init, pi_zero_init, sigma_init, _sigma_zero = cluster_weights(init_weights, self.num_components)
            sigma_init, _sigma_zero = torch.ones_like(sigma_init).mul(0.01).npu(), torch.ones_like(torch.tensor([_sigma_zero])).mul(0.01).npu()
        self.mu = nn.Parameter(data=torch.mul(self.mu.npu(), initial_region_saliency.flatten().npu()))
        self.pi_k = nn.Parameter(data=torch.mul(self.pi_k.npu(), pi_init)).npu().float()
        self.pi_zero = nn.Parameter(data=torch.tensor([pi_zero_init], device=self.device)).npu().float()
        self.sigma_zero = nn.Parameter(data=torch.tensor([_sigma_zero], device=self.device)).float()
        self.sigma = nn.Parameter(data=torch.mul(self.sigma, sigma_init)).npu().float()
        self.temperature = nn.Parameter(data=torch.tensor([self.temperature], device=self.device))

    def gaussian_mixing_regularization(self):
        pi_tmp = torch.cat([self.pi_zero, self.pi_k], dim=-1).abs()
        return torch.div(pi_tmp, pi_tmp.sum(dim=-1).unsqueeze(-1)).npu()

    def Normal_pdf(self, x, _pi, mu, sigma):
        """ Standard Normal Distribution PDF. """
        return torch.mul(torch.reciprocal(torch.sqrt(torch.mul( \
               torch.tensor([2 * math.pi], device=self.device), sigma**2))), \
               torch.exp(-torch.div((x - mu)**2, 2 * sigma**2))).mul(_pi)

    def GMM_region_responsibility(self, weights):
        """" Region responsibility of GMM. """
        pi_normalized = self.gaussian_mixing_regularization().npu()
        responsibility = torch.zeros([self.num_components, weights.size(0)], device=self.device)
        responsibility[0] = self.Normal_pdf(weights.npu(), pi_normalized[0], 0.0, self.sigma_zero.npu())
        for k in range(self.num_components-1):
            responsibility[k+1] = self.Normal_pdf(weights, pi_normalized[k+1], self.mu[k].npu(), self.sigma[k].npu())
        responsibility = torch.div(responsibility, responsibility.sum(dim=0) + cfg.EPS)
        return F.softmax(responsibility / self.temperature, dim=0)

    def forward(self, weights, train=True):
        if train:
            # soft mask generalized pruning during training
            self.region_belonging = self.GMM_region_responsibility(weights.flatten())
            Sweight = torch.mul(self.region_belonging[0], 0.) \
                    + torch.mul(self.region_belonging[1:], self.mu.unsqueeze(1)).sum(dim=0)
            return Sweight.view(weights.size())
        else:
            self.region_belonging = self.GMM_region_responsibility(weights.flatten())
            max_index = torch.argmax(self.region_belonging, dim=0).unsqueeze(0)
            mask_w = torch.zeros_like(self.region_belonging).scatter_(dim=0, index=max_index, value=1.)
            Pweight = torch.mul(mask_w[1:], self.mu.unsqueeze(1)).sum(dim=0)
            return Pweight.view(weights.size())

def gmm_approximation(num_components, init_weights, temperature=0.5, init_method='k-means'):
    return GaussianMixtureModel(num_components, init_weights.flatten(), temperature, init_method)
