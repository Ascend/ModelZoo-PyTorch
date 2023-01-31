# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# coding: utf-8
import argparse
import time
import math
import os
import sys
import torch

parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')
parser.add_argument('--data', type=str, default='../data/enwik8',
                    help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='enwik8',
                    choices=['wt103', 'lm1b', 'enwik8', 'text8'],
                    help='dataset name')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch size')
parser.add_argument('--tgt_len', type=int, default=128,
                    help='number of tokens to predict')
parser.add_argument('--ext_len', type=int, default=0,
                    help='length of the extended context')
parser.add_argument('--mem_len', type=int, default=0,
                    help='length of the retained previous heads')
parser.add_argument('--clamp_len', type=int, default=-1,
                    help='max positional embedding index')
parser.add_argument('--cuda', default=False, action='store_true',
                    help='use CUDA')
parser.add_argument('--work_dir', type=str,
                    help='path to the work_dir')
parser.add_argument('--same_length', action='store_true', default=True,
                    help='set same length attention with masking')


args = parser.parse_args()
assert args.ext_len >= 0, 'extended context length must be non-negative'
device = torch.device("cuda" if args.cuda else "cpu")

# Load the best saved model.
with open(os.path.join(args.work_dir, 'model.pt'), 'rb') as f:
    model = torch.load(f, map_location=device)
model.backward_compatible()
model = model.to(device)

print('Evaluating with bsz {} tgt_len {} ext_len {} mem_len {} clamp_len {}'.format(
       args.batch_size, args.tgt_len, args.ext_len, args.mem_len, args.clamp_len))

model.reset_length(args.tgt_len, args.ext_len, args.mem_len)
if args.clamp_len > 0:
    model.clamp_len = args.clamp_len
if args.same_length:
    model.same_length = True

# export onnx model
data = torch.ones(args.tgt_len, args.batch_size, dtype=torch.int64).to(device)
target = torch.ones(args.tgt_len, args.batch_size, dtype=torch.int64).to(device)
model.eval()
mems = tuple()
ret = model(data, target, *mems)
loss, mems = ret[0], ret[1:]
loss = loss.mean()
print('*'*100)
onnx_name = "model_bs" + str(args.batch_size) + ".onnx"
torch.onnx.export(model, (data, target, *mems), onnx_name, input_names=['data', 'target'], output_names=['output'],
                  do_constant_folding=True, keep_initializers_as_inputs=True, opset_version=12, verbose=True)
print("export onnx model success")
sys.exit()
