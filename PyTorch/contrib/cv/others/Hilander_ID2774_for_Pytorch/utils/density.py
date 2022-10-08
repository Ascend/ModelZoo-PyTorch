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
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file re-uses implementation from https://github.com/yl-1993/learn-to-cluster
"""

import numpy as np
from tqdm import tqdm
from itertools import groupby
import torch

__all__ = ['density_estimation', 'density_to_peaks', 'density_to_peaks_vectorize']

def density_estimation(dists, nbrs, labels, **kwargs):
    ''' use supervised density defined on neigborhood
    '''
    num, k_knn = dists.shape
    conf = np.ones((num, ), dtype=np.float32)
    ind_array = labels[nbrs] == np.expand_dims(labels, 1).repeat(k_knn, 1)
    pos = ((1-dists[:,1:]) * ind_array[:,1:]).sum(1)
    neg = ((1-dists[:,1:]) * (1-ind_array[:,1:])).sum(1)
    conf = (pos - neg) * conf
    conf /= (k_knn - 1)
    return conf

def density_to_peaks_vectorize(dists, nbrs, density, max_conn=1, name = ''):
    # just calculate 1 connectivity
    assert dists.shape[0] == density.shape[0]
    assert dists.shape == nbrs.shape

    num, k = dists.shape

    if name == 'gcn_feat':
        include_mask = nbrs != np.arange(0, num).reshape(-1, 1)
        secondary_mask = np.sum(include_mask, axis = 1) == k # TODO: the condition == k should not happen as distance to the node self should be smallest, check for numerical stability; TODO: make top M instead of only supporting top 1
        include_mask[secondary_mask, -1] = False
        nbrs_exclude_self = nbrs[include_mask].reshape(-1, k-1) # (V, 79)
        dists_exclude_self = dists[include_mask].reshape(-1, k-1) # (V, 79)
    else:
        include_mask = nbrs != np.arange(0, num).reshape(-1, 1)
        nbrs_exclude_self = nbrs[include_mask].reshape(-1, k-1) # (V, 79)
        dists_exclude_self = dists[include_mask].reshape(-1, k-1) # (V, 79)

    compare_map = density[nbrs_exclude_self] > density.reshape(-1, 1)
    peak_index = np.argmax(np.where(compare_map, 1, 0), axis = 1) # (V,)
    compare_map_sum = np.sum(compare_map.cpu().data.numpy(), axis=1) # (V,)

    dist2peak = {i: [] if compare_map_sum[i] == 0 else [dists_exclude_self[i, peak_index[i]]] for i in range(num)}
    peaks = {i: [] if compare_map_sum[i] == 0 else [nbrs_exclude_self[i, peak_index[i]]] for i in range(num)}

    return dist2peak, peaks

def density_to_peaks(dists, nbrs, density, max_conn=1, sort='dist'):
    # Note that dists has been sorted in ascending order
    assert dists.shape[0] == density.shape[0]
    assert dists.shape == nbrs.shape

    num, _ = dists.shape
    dist2peak = {i: [] for i in range(num)}
    peaks = {i: [] for i in range(num)}

    for i, nbr in tqdm(enumerate(nbrs)):
        nbr_conf = density[nbr]
        for j, c in enumerate(nbr_conf):
            nbr_idx = nbr[j]
            if i == nbr_idx or c <= density[i]:
                continue
            dist2peak[i].append(dists[i, j])
            peaks[i].append(nbr_idx)
            if len(dist2peak[i]) >= max_conn:
                break

    return dist2peak, peaks
