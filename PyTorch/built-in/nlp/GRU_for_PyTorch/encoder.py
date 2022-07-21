# Copyright [yyyy] [name of copyright owner]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
if torch.__version__ >= "1.8":
    import torch_npu
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout, seed=0):
        super().__init__()

        self.hid_dim = hid_dim
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

        self.seed = seed
        self.prob = dropout

    def forward(self, src):
        embedded = self.embedding(src)

        if self.training:
            if torch.__version__ >= "1.8.1":
                embedded = nn.functional.dropout(embedded, p=self.prob)
            else:
                embedded, _, _ = torch.npu_dropoutV2(embedded, self.seed, p=self.prob)

        outputs, hidden = self.rnn(embedded)  # no cell state!

        return hidden
