# coding:utf-8
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


import argparse
from train import train
from evaluate import evaluator, t2vec
import torch
import torch_npu


from torch.nn.utils.rnn import PackedSequence


def gru_forward(self, input_tensor, hx=None):
    orig_input = input_tensor

    if isinstance(orig_input, PackedSequence):
        input_tensor, batch_sizes, sorted_indices, unsorted_indices = input_tensor
        max_batch_size = batch_sizes[0]
        max_batch_size = int(max_batch_size)
    else:
        batch_sizes = None
        max_batch_size = input_tensor.size(0) if self.batch_first else input_tensor.size(1)
        sorted_indices = None
        unsorted_indices = None

    if hx is None:
        num_directions = 2 if self.bidirectional else 1
        hx = torch.zeros(self.num_layers * num_directions,
                         max_batch_size, self.hidden_size,
                         dtype=input_tensor.dtype, device=input_tensor.device)
    else:
        # Each batch of the hidden state should match the input sequence that
        # the user believes he/she is passing in.
        hx = self.permute_hidden(hx, sorted_indices)

    self.check_forward_args(input_tensor, hx, batch_sizes)
    if batch_sizes is None:
        result = torch._VF.gru(input_tensor, hx, self._flat_weights, self.bias, self.num_layers,
                               self.dropout, self.training, self.bidirectional, self.batch_first)
    else:
        if batch_sizes.device != input_tensor.device:
            # convert to compact length
            start = 0
            idx_list = []
            batch_list = batch_sizes.numpy()
            for i in batch_list:
                idx = list(range(start, start + i, 1))
                idx_list = idx_list + idx
                start = start + batch_list[0]
            input_pack = input_tensor
            if len(idx_list) != input_tensor.shape[0]:
                idx_tensor = torch.Tensor(idx_list).long().to(input_tensor.device)
                input_pack = torch.nn.functional.embedding(idx_tensor, input_tensor)

            result = torch._VF.gru(input_pack, batch_sizes, hx, self._flat_weights, self.bias,
                                   self.num_layers, self.dropout, self.training, self.bidirectional)

            # convert to fixed length
            if len(idx_list) != input_tensor.shape[0]:
                start = 0
                cur = start
                shape = [1] + list(result[0].shape[1:])
                pad_tensor = torch.zeros(shape, device = "cpu")
                cat_list = []
                for ii, i in enumerate(batch_list):#128
                    if (i < batch_list[0]):
                        slice_tensor = result[0][start : cur + i, :]
                        start = cur + i
                        cur = start
                        cat_list.append(slice_tensor.cpu())
                        for j in range(batch_list[0] - i):
                            cat_list.append(pad_tensor)
                    else:
                        cur = cur + batch_list[0]
                result0 = torch.cat(cat_list, 0).npu(non_blocking=True)
                result = (result0, result[1])
        else:
            result = torch._VF.gru(input_tensor, batch_sizes, hx, self._flat_weights, self.bias,
                                   self.num_layers, self.dropout, self.training, self.bidirectional)
    output = result[0]
    hidden = result[1]

    if isinstance(orig_input, PackedSequence):
        output_packed = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
        return output_packed, self.permute_hidden(hidden, unsorted_indices)
    else:
        return output, self.permute_hidden(hidden, unsorted_indices)

torch.nn.modules.rnn.GRU.forward = gru_forward


parser = argparse.ArgumentParser(description="train.py")

parser.add_argument("-data", default="./data",
    help="Path to training and validating data")

parser.add_argument("-checkpoint", default="./data/checkpoint.pt",
    help="The saved checkpoint")

parser.add_argument("-prefix", default="exp", help="Prefix of trjfile")

parser.add_argument("-pretrained_embedding", default=None,
    help="Path to the pretrained word (cell) embedding")

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

parser.add_argument("-max_grad_norm", type=float, default=5.0,
    help="The maximum gradient norm")

parser.add_argument("-learning_rate", type=float, default=0.001)

parser.add_argument("-batch", type=int, default=128,
    help="The batch size")

parser.add_argument("-generator_batch", type=int, default=32,
    help="""The maximum number of words to generate each time.
    The higher value, the more memory requires.""")

parser.add_argument("-t2vec_batch", type=int, default=256,
    help="""The maximum number of trajs we encode each time in t2vec""")

parser.add_argument("-start_iteration", type=int, default=0)

parser.add_argument("-epochs", type=int, default=15,
    help="The number of training epochs")

parser.add_argument("-print_freq", type=int, default=50,
    help="Print frequency")

parser.add_argument("-save_freq", type=int, default=1000,
    help="Save frequency")

parser.add_argument("-npu", type=bool, default=True,
    help="True if we use NPU to train the model")

parser.add_argument("-use_discriminative", action="store_true",
    help="Use the discriminative loss if the argument is given")

parser.add_argument("-discriminative_w", type=float, default=0.1,
    help="discriminative loss weight")

parser.add_argument("-criterion_name", default="NLL",
    help="NLL (Negative Log Likelihood) or KLDIV (KL Divergence)")

parser.add_argument("-knearestvocabs", default=None,
    help="""The file of k nearest cells and distances used in KLDIVLoss,
    produced by preprocessing, necessary if KLDIVLoss is used""")

parser.add_argument("-dist_decay_speed", type=float, default=0.8,
    help="""How fast the distance decays in dist2weight, a small value will
    give high weights for cells far away""")

parser.add_argument("-max_num_line", type=int, default=20000000)

parser.add_argument("-max_length", default=200,
    help="The maximum length of the target sequence")

parser.add_argument("-mode", type=int, default=0,
    help="Running mode (0: train, 1:evaluate, 2:t2vec)")

parser.add_argument("-vocab_size", type=int, default=0,
    help="Vocabulary Size")

parser.add_argument("-local_rank", type=int, default=0,
    help="Vocabulary Size")

parser.add_argument("-max_step", type=int, default=0,
    help="Vocabulary Size")

parser.add_argument("-bucketsize", default=[(20,30),(30,30),(30,50),(50,50),(50,70),(70,70),(70,100),(100,100)],
    help="Bucket size for training")



args = parser.parse_args()

print(args)

## __main__
#args.bucketsize = [(20,30),(30,30),(30,50),(50,50),(50,70),(70,70),(70,100),(100,100)]
#args.bucketsize = [(10, 10), (20, 20), (20, 30)]
#args.vocab_size = 43

if args.mode == 1:
    evaluator(args)
elif args.mode == 2:
    with torch.no_grad():
        t2vec(args)
else:
    train(args)
