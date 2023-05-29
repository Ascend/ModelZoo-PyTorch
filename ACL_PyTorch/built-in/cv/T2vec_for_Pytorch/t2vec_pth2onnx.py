# Copyright 2023 Huawei Technologies Co., Ltd
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
import os

import torch
import numpy as np

from models import EncoderDecoder 


def parse_args():
    parser = argparse.ArgumentParser(description="Configs")

    parser.add_argument("-output_file", default="t2vec.onnx",
        help="Path of outputfile")
    
    parser.add_argument("-prep_data", default="./prep_data",
        help="Path to data after preprocessing")

    parser.add_argument("-checkpoint", default="./data/best_model.pt",
        help="The saved checkpoint")

    parser.add_argument("-num_layers", type=int, default=3,
        help="Number of layers in the RNN cell")

    parser.add_argument("-bidirectional", type=bool, default=True,
        help="True if use bidirectional rnn in encoder")

    parser.add_argument("-hidden_size", type=int, default=256,
        help="The hidden state size in the RNN cell")

    parser.add_argument("-embedding_size", type=int, default=256,
        help="The word (cell) embedding size")

    parser.add_argument("-dropout", type=float, default=0.2,
        help="The dropout probability")

    parser.add_argument("-t2vec_batch", type=int, default=256,
        help="""The maximum number of trajs we encode each time in t2vec""")

    parser.add_argument("-vocab_size", type=int, default=18866,
        help="Vocabulary Size")
    
    args = parser.parse_args()
    return args


def pth2onnx(args):
    m0 = EncoderDecoder(args.vocab_size, args.embedding_size,
                        args.hidden_size, args.num_layers,
                        args.dropout, args.bidirectional)
    checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    m0.load_state_dict(checkpoint["m0"])
    m0.eval()
    
    src_path = os.path.join(args.prep_data, "src/1.npy")
    lengths_path = os.path.join(args.prep_data, "lengths/1.npy")
    src = torch.from_numpy(np.load(src_path))
    lengths = torch.from_numpy(np.load(lengths_path))
    
    input_data = (src, lengths)
    input_names = ['src', 'lengths']
    output_names = ['h']
    dynamic_axes = {
        'src': {0: 'seq_len'},
        }
    torch.onnx.export(
        m0,
        input_data,
        args.output_file,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        verbose=True,
        opset_version=11,
    )


if __name__ == "__main__":
    opt = parse_args()
    pth2onnx(opt)
