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
import os
import sys

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')

def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.
        
    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """

    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).npu()
    index = indices.view(indices.size()+torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1,index,1)
    
    return encoded_indicies

def CosineNetHead(k_all, meta_part_infer, query, support, support_labels, n_way, n_shot, is_scale=False, normalize=True):
    """
    Constructs the prototype representation of each class(=mean of support vectors of each class) and
    returns the classification score (=L2 distance to each class prototype) on the query set.

    This model is the classification head described in:
    Prototypical Networks for Few-shot Learning
    (Snell et al., NIPS 2017).

    Parameters:
      query:  a (tasks_per_batch, n_query, d) Tensor.
      support:  a (tasks_per_batch, n_support, d) Tensor.
      support_labels: a (tasks_per_batch, n_support) Tensor.
      n_way: a scalar. Represents the number of classes in a few-shot classification task.
      n_shot: a scalar. Represents the number of support examples given per class.
      normalize: a boolean. Represents whether if we want to normalize the distances by the embedding dimension.
    Returns: a (tasks_per_batch, n_query, n_way) Tensor.
    """

    tasks_per_batch = query.size(0)
    n_support = support.size(1)
    n_query = query.size(1)
    d = query.size(2)

    assert (query.dim() == 3)
    assert (support.dim() == 3)
    assert (query.size(0) == support.size(0) and query.size(2) == support.size(2))
    assert (n_support == n_way * n_shot)  # n_support must equal to n_way * n_shot

    support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * n_support), n_way)
    support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, n_support, n_way)

    # From:
    # https://github.com/gidariss/FewShotWithoutForgetting/blob/master/architectures/PrototypicalNetworksHead.py
    # ************************* Compute Prototypes **************************
    labels_train_transposed = support_labels_one_hot.transpose(1, 2)

    prototypes = torch.bmm(labels_train_transposed, support)
    # Divide with the number of examples per novel category.
    prototypes = prototypes.div(
        labels_train_transposed.sum(dim=2, keepdim=True).expand_as(prototypes)
    )

    boost_prototypes, _ = meta_part_infer(prototypes.reshape(-1, d), k_all.reshape(-1), is_infer=is_scale)
    boost_prototypes = boost_prototypes.reshape(tasks_per_batch, n_way, d)
    prototypes = prototypes * 0.5 + boost_prototypes * 0.5
    # Distance Matrix Vectorization Trick
    logits = torch.nn.functional.cosine_similarity(query.unsqueeze(2).expand(-1, -1, prototypes.shape[1], -1),
                                                   prototypes.unsqueeze(1).expand(-1, query.shape[1], -1, -1), dim=-1)

    return logits


def FuseCosineNetHead(k_all, meta_part_infer, query, support, support_labels, n_way, n_shot, is_scale=False, normalize=True):
    """
    Constructs the prototype representation of each class(=mean of support vectors of each class) and
    returns the classification score (=L2 distance to each class prototype) on the query set.

    This model is the classification head described in:
    Prototypical Networks for Few-shot Learning
    (Snell et al., NIPS 2017).

    Parameters:
      query:  a (tasks_per_batch, n_query, d) Tensor.
      support:  a (tasks_per_batch, n_support, d) Tensor.
      support_labels: a (tasks_per_batch, n_support) Tensor.
      n_way: a scalar. Represents the number of classes in a few-shot classification task.
      n_shot: a scalar. Represents the number of support examples given per class.
      normalize: a boolean. Represents whether if we want to normalize the distances by the embedding dimension.
    Returns: a (tasks_per_batch, n_query, n_way) Tensor.
    """
    scale = 10
    tasks_per_batch = query.size(0)
    n_support = support.size(1)
    n_query = query.size(1)
    d = query.size(2)

    assert (query.dim() == 3)
    assert (support.dim() == 3)
    assert (query.size(0) == support.size(0) and query.size(2) == support.size(2))
    assert (n_support == n_way * n_shot)  # n_support must equal to n_way * n_shot

    support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * n_support), n_way)
    support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, n_support, n_way)

    # From:
    # https://github.com/gidariss/FewShotWithoutForgetting/blob/master/architectures/PrototypicalNetworksHead.py
    # ************************* Compute Prototypes **************************
    labels_train_transposed = support_labels_one_hot.transpose(1, 2)

    prototypes = torch.bmm(labels_train_transposed, support)
    # Divide with the number of examples per novel category.
    prototypes = prototypes.div(
        labels_train_transposed.sum(dim=2, keepdim=True).expand_as(prototypes)
    )
    if is_scale:
        boost_prototypes, _ = meta_part_infer(prototypes.reshape(-1, d), k_all.reshape(-1), use_scale=is_scale, is_infer=is_scale)
        boost_prototypes = boost_prototypes.reshape(tasks_per_batch, n_way, d)
    else:
        boost_prototypes = meta_part_infer(prototypes.reshape(-1, d), k_all.reshape(-1), use_scale=is_scale, is_infer=is_scale)
        boost_prototypes = boost_prototypes[0].reshape(tasks_per_batch, n_way, d)

    logits = torch.nn.functional.cosine_similarity(query.unsqueeze(2).expand(-1, -1, prototypes.shape[1], -1),
                                                   prototypes.unsqueeze(1).expand(-1, query.shape[1], -1, -1), dim=-1)
    assign_1 = F.softmax(logits * scale, dim=-1)
    assign_1 = torch.cat([support_labels_one_hot, assign_1], dim=1)
    assign_1_transposed = assign_1.transpose(1, 2)
    emb = torch.cat([support, query], dim=1)
    mean_1 = torch.bmm(assign_1_transposed, emb)
    mean_1 = mean_1.div(
        assign_1_transposed.sum(dim=2, keepdim=True).expand_as(mean_1)
    )
    diff = torch.pow(emb.unsqueeze(1).expand(-1, n_way, -1, -1) - mean_1.unsqueeze(2).expand(-1, -1, emb.shape[1], -1), 2)
    std_1 = (assign_1_transposed.unsqueeze(-1).expand_as(diff) * diff).sum(dim=2) / assign_1_transposed.unsqueeze(-1).expand_as(diff).sum(dim=2)

    logits = torch.nn.functional.cosine_similarity(query.unsqueeze(2).expand(-1, -1, boost_prototypes.shape[1], -1),
                                                   boost_prototypes.unsqueeze(1).expand(-1, query.shape[1], -1, -1), dim=-1)
    assign_2 = F.softmax(logits * scale, dim=-1)
    assign_2 = torch.cat([support_labels_one_hot, assign_2], dim=1)
    assign_2_transposed = assign_2.transpose(1, 2)
    emb = torch.cat([support, query], dim=1)
    mean_2 = torch.bmm(assign_2_transposed, emb)
    mean_2 = mean_2.div(
        assign_2_transposed.sum(dim=2, keepdim=True).expand_as(mean_2)
    )
    diff = torch.pow(emb.unsqueeze(1).expand(-1, n_way, -1, -1) - mean_2.unsqueeze(2).expand(-1, -1, emb.shape[1], -1), 2)
    std_2 = (assign_2_transposed.unsqueeze(-1).expand_as(diff) * diff).sum(dim=2) / assign_2_transposed.unsqueeze(-1).expand_as(diff).sum(dim=2)

    prototypes = (mean_1 * std_2 + mean_2 * std_1) / (std_2 + std_1)
    logits = torch.nn.functional.cosine_similarity(query.unsqueeze(2).expand(-1, -1, prototypes.shape[1], -1),
                                                   prototypes.unsqueeze(1).expand(-1, query.shape[1], -1, -1), dim=-1)
    # Distance Matrix Vectorization Trick
    return logits

class ClassificationHead(nn.Module):
    def __init__(self, base_learner='MetaOptNet', enable_scale=True):
        super(ClassificationHead, self).__init__()
        if ('Cosine' in base_learner):
            self.head = CosineNetHead
        elif ('FuseCos' in base_learner):
            self.head = FuseCosineNetHead
        else:
            print ("Cannot recognize the base learner type")
            assert(False)
        
        # Add a learnable scale
        self.enable_scale = enable_scale
        self.scale = nn.Parameter(torch.FloatTensor([1.0]))
        
    def forward(self, k_all, meta_part_infer, query, support, support_labels, n_way, n_shot, **kwargs):
        if self.enable_scale:
            return self.scale * self.head(k_all, meta_part_infer, query, support, support_labels, n_way, n_shot, **kwargs)
        else:
            return self.head(k_all, meta_part_infer, query, support, support_labels, n_way, n_shot, **kwargs)