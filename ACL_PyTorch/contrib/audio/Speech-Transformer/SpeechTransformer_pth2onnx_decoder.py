# Copyright 2021 Huawei Technologies Co., Ltd
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
import argparse
import torch
import torch.nn as nn
from collections import OrderedDict

import sys
sys.path.append('../../src/utils/')
sys.path.append('../../src/transformer/')
from module import PositionalEncoding
from utils import get_subsequent_mask
from decoder import DecoderLayer


parser = argparse.ArgumentParser("Speech-Transformer-pth2onnx-decoder")
parser.add_argument('--pth-path', type=str,  default='./final.pth.tar')
parser.add_argument('--decoder-path', type=str, default='./decoder.onnx')


def pth2onnx(model, output_file, input):
    model = model.to('cpu')
    model.eval()
    input_names = ["ys_in", 'encoder_outputs', "non_pad_mask", "slf_attn_mask"]
    output_names = ["dec_output"]
    torch.onnx.export(model, input, output_file, input_names=input_names,
                      output_names=output_names, opset_version=11, verbose=True)


def proc_node_module(checkpoint, attr_name):
    new_state_dict = OrderedDict()
    for k, v in checkpoint[attr_name].items():
        if(k[0: 8] == "decoder."):
            name = k[8:]
        else:
            name = k[0:]
        new_state_dict[name] = v
    return new_state_dict


def load_model_from_package(path):
    package = torch.load(path, map_location=lambda storage, loc: storage)
    decoder = Decoder(package['vocab_size'],
                      package['d_word_vec'],
                      package['n_layers_dec'],
                      package['n_head'],
                      package['d_k'],
                      package['d_v'],
                      package['d_model'],
                      package['d_inner'],
                      dropout=package['dropout'],
                      tgt_emb_prj_weight_sharing=package['tgt_emb_prj_weight_sharing'],
                      pe_maxlen=package['pe_maxlen'],
                      )
    transformer_state_dict = proc_node_module(package, 'state_dict')
    decoder.load_state_dict(transformer_state_dict, strict=False)
    return decoder


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, n_tgt_vocab, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1,
            tgt_emb_prj_weight_sharing=True,
            pe_maxlen=5000):
        super(Decoder, self).__init__()
        # parameters`
        self.n_tgt_vocab = n_tgt_vocab
        self.d_word_vec = d_word_vec
        self.n_layers = n_layers
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout
        self.tgt_emb_prj_weight_sharing = tgt_emb_prj_weight_sharing
        self.pe_maxlen = pe_maxlen

        self.tgt_word_emb = nn.Embedding(n_tgt_vocab, d_word_vec)
        self.positional_encoding = PositionalEncoding(
            d_model, max_len=pe_maxlen)
        self.dropout = nn.Dropout(dropout)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

        self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.tgt_word_prj.weight = self.tgt_word_emb.weight
            self.x_logit_scale = (d_model ** 0.5)
        else:
            self.x_logit_scale = 1.

    def forward(self, ys, encoder_outputs, non_pad_mask, slf_attn_mask):
        dec_output = self.tgt_word_emb(
            ys) * self.x_logit_scale + self.positional_encoding(ys)

        for dec_layer in self.layer_stack:
            dec_output, _, _ = dec_layer(
                dec_output, encoder_outputs,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=None)
        return dec_output


def main():
    args = parser.parse_args()
    model_path = args.pth_path
    output_path = args.decoder_path
    model = load_model_from_package(model_path).to('cpu')
    model.eval()
    dummy_ys = torch.zeros([1, 128]).to('cpu').long()
    dummy_ys[0, 0] = 1
    dummy_encoder_outputs = torch.randn([1, 512, 512])
    non_pad_mask = torch.ones_like(dummy_ys).float().unsqueeze(-1)
    slf_attn_mask = get_subsequent_mask(dummy_ys)
    input = (dummy_ys, dummy_encoder_outputs, non_pad_mask, slf_attn_mask)
    pth2onnx(model, output_path, input)


if __name__ == "__main__":
    main()
