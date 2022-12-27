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

import torch

from transformers import AutoModel


class FSNERModel(torch.nn.Module):
    """
    The FSNER model implements a few-shot named entity recognition method from the paper `Example-Based Named Entity Recognition <https://arxiv.org/abs/2008.10570>`__ by
    Morteza Ziyadi, Yuting Sun, Abhishek Goswami, Jade Huang, Weizhu Chen. To identify entity spans in a new domain, it
    uses a train-free few-shot learning approach inspired by question-answering.
    """

    def __init__(self, pretrained_model_name_or_path="sayef/fsner-bert-base-uncased"):
        super(FSNERModel, self).__init__()

        self.bert = AutoModel.from_pretrained(pretrained_model_name_or_path, return_dict=True)
        self.cos = torch.nn.CosineSimilarity(3, 1e-08)
        self.softmax = torch.nn.Softmax(dim=1)

    def BERT(self, **inputs):
        return self.bert(**inputs).last_hidden_state

    def VectorSum(self, token_embeddings):
        return token_embeddings.sum(2, keepdim=True)

    def Atten(self, q_rep, S_rep, T=1):
        return self.softmax(T * self.cos(q_rep, S_rep))

    def forward(self, W_query, W_supports):
        """
        Find scores of each token being start and end token for an entity.
        Args:
            W_query (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of query sequence tokens in the vocabulary.
            W_supports (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of support sequence tokens in the vocabulary.
        Returns:
            p_start (`torch.FloatTensor` of shape `(batch_size, sequence_length)`): Scores of each token as
            being start token of an entity
            p_end (`torch.FloatTensor` of shape `(batch_size, sequence_length)`): Scores of each token as
            being end token of an entity
        """

        support_sizes = W_supports["sizes"].tolist()
        start_token_id = W_supports["start_token_id"].item()
        end_token_id = W_supports["end_token_id"].item()

        del W_supports["sizes"]
        del W_supports["start_token_id"]
        del W_supports["end_token_id"]

        q = self.BERT(**W_query)
        S = self.BERT(**W_supports)

        p_starts = None
        p_ends = None

        start_token_masks = W_supports["input_ids"] == start_token_id
        end_token_masks = W_supports["input_ids"] == end_token_id

        for i, size in enumerate(support_sizes):
            if i == 0:
                s = 0
            else:
                s = support_sizes[i - 1]

            s_start = S[s : s + size][start_token_masks[s : s + size]]
            s_end = S[s : s + size][end_token_masks[s : s + size]]

            p_start = torch.matmul(q[i], s_start.T).sum(1).softmax(0)
            p_end = torch.matmul(q[i], s_end.T).sum(1).softmax(0)

            if p_starts is not None:
                p_starts = torch.vstack((p_starts, p_start))
                p_ends = torch.vstack((p_ends, p_end))
            else:
                p_starts = p_start
                p_ends = p_end

        return p_starts, p_ends
