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

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from bert4torch.snippets import sequence_padding, Callback, ListDataset, seed_everything
from bert4torch.losses import FocalLoss
from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model, BaseModel, BERT
from tqdm import tqdm


config_path = 'F:/Projects/pretrain_ckpt/uie/uie_base_pytorch/config.json'
checkpoint_path = 'F:/Projects/pretrain_ckpt/uie/uie_base_pytorch/pytorch_model.bin'
dict_path = 'F:/Projects/pretrain_ckpt/uie/uie_base_pytorch/vocab.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = Tokenizer(dict_path, do_lower_case=True)


class UIE(BERT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_size = self.hidden_size

        self.linear_start = nn.Linear(hidden_size, 1)
        self.linear_end = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

        if kwargs.get('use_task_id') and kwargs.get('use_task_id'):
            # Add task type embedding to BERT
            task_type_embeddings = nn.Embedding(kwargs.get('task_type_vocab_size'), self.hidden_size)
            self.embeddings.task_type_embeddings = task_type_embeddings

            def hook(module, input, output):
                return output+task_type_embeddings(torch.zeros(input[0].size(), dtype=torch.int64, device=input[0].device))
            self.embeddings.word_embeddings.register_forward_hook(hook)

    def forward(self, token_ids, token_type_ids):
        outputs = super().forward([token_ids, token_type_ids])
        sequence_output = outputs[0]

        start_logits = self.linear_start(sequence_output)
        start_logits = torch.squeeze(start_logits, -1)
        start_prob = self.sigmoid(start_logits)
        end_logits = self.linear_end(sequence_output)
        end_logits = torch.squeeze(end_logits, -1)
        end_prob = self.sigmoid(end_logits)

        return start_prob, end_prob
    
    @torch.no_grad()
    def predict(self, token_ids, token_type_ids):
        self.eval()
        start_prob, end_prob = self.forward(token_ids, token_type_ids)
        return start_prob, end_prob

uie_model = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, model=UIE, with_pool=True)