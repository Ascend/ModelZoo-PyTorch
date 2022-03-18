import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout, seed=0):
        super().__init__()

        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim + hid_dim, hid_dim)
        self.fc_out = nn.Linear(emb_dim + hid_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

        self.seed = seed
        self.prob = dropout

    def forward(self, input, hidden, context):
        input = input.unsqueeze(0)
        embedded = self.embedding(input)
        if self.training:
            embedded, _, _ = torch.npu_dropoutV2(embedded, self.seed, p=self.prob)

        emb_con = torch.cat((embedded, context), dim=2)
        output, hidden = self.rnn(emb_con, hidden)
        output = torch.cat((embedded.squeeze(0), hidden.squeeze(0), context.squeeze(0)),
                           dim=1)
        prediction = self.fc_out(output)

        return prediction, hidden
