import torch
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
            embedded, _, _ = torch.npu_dropoutV2(embedded, self.seed, p=self.prob)

        outputs, hidden = self.rnn(embedded)  # no cell state!

        return hidden
