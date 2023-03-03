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
import sys
import torch
import torch.nn as nn
from collections import OrderedDict

sys.path.append('../../src/utils/')
sys.path.append('../../src/transformer/')
from module import PositionalEncoding
from utils import (get_attn_pad_mask, get_non_pad_mask,  pad_list)
from encoder import EncoderLayer

parser = argparse.ArgumentParser("Speech-Transformer-pth2onnx-encoder")
parser.add_argument('--pth-path', type=str,  default='./final.pth.tar')
parser.add_argument('--encoder-path', type=str, default='./encoder.onnx')


def pth2onnx(model, output_file, input):
    model = model.to('cpu')
    model.eval()
    input_names = ["padded_input", "non_pad_mask", "slf_attn_mask"]
    output_names = ["enc_output"]
    torch.onnx.export(model, input, output_file, input_names=input_names,
                      output_names=output_names, opset_version=11, verbose=True)


def proc_node_module(checkpoint, attr_name):
    new_state_dict = OrderedDict()
    for k, v in checkpoint[attr_name].items():
        if(k[0: 8] == "encoder."):
            name = k[8:]
        else:
            name = k[0:]
        new_state_dict[name] = v
    return new_state_dict


class Encoder(nn.Module):
    """Encoder of Transformer including self-attention and feed forward.
    """

    def __init__(self, d_input, n_layers, n_head, d_k, d_v,
                 d_model, d_inner, dropout=0.1, pe_maxlen=5000):
        super(Encoder, self).__init__()
        # parameters
        self.d_input = d_input
        self.n_layers = n_layers
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout_rate = dropout
        self.pe_maxlen = pe_maxlen

        # use linear transformation with layer norm to replace input embedding
        self.linear_in = nn.Linear(d_input, d_model)
        self.layer_norm_in = nn.LayerNorm(d_model)
        self.positional_encoding = PositionalEncoding(
            d_model, max_len=pe_maxlen)
        self.dropout = nn.Dropout(dropout)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, padded_input, non_pad_mask, slf_attn_mask, return_attns=False):
        """
        Args:
            padded_input: N x T x D

        Returns:
            enc_output: N x T x H
        """
        enc_slf_attn_list = []
        # Forward
        enc_output = self.dropout(
            self.layer_norm_in(self.linear_in(padded_input)) +
            self.positional_encoding(padded_input))

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


def load_model_from_package(path):
    package = torch.load(path, map_location=lambda storage, loc: storage)
    encoder = Encoder(package['d_input'],
                      package['n_layers_enc'],
                      package['n_head'],
                      package['d_k'],
                      package['d_v'],
                      package['d_model'],
                      package['d_inner'],
                      dropout=package['dropout'],
                      pe_maxlen=package['pe_maxlen'])
    transformer_state_dict = proc_node_module(package, 'state_dict')
    encoder.load_state_dict(transformer_state_dict, strict=False)
    return encoder


def main():
    args = parser.parse_args()
    model_path = args.pth_path
    output_path = args.encoder_path
    model = load_model_from_package(model_path).to('cpu')
    model.eval()
    # Prepare masks
    padded_input = torch.rand(size=(1, 140, 320), dtype=torch.float32)
    input_lengths = torch.tensor([padded_input.size(1)], dtype=torch.int)
    padded_input = pad_list(padded_input, 0, max_len=512)
    non_pad_mask = get_non_pad_mask(padded_input, input_lengths=input_lengths)
    length = padded_input.size(1)
    slf_attn_mask = get_attn_pad_mask(padded_input, input_lengths, length)

    input = (padded_input, non_pad_mask, slf_attn_mask)
    pth2onnx(model, output_path, input)


if __name__ == "__main__":
    main()
