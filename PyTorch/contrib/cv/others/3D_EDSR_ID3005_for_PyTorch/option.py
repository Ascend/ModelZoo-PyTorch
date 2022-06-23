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

parser = argparse.ArgumentParser(description='3D EDSR Training in Micro-CT Rock Images With Pytorch')

# Hardware specifications
parser.add_argument('--threads', type=int, default=1,
                    help='number of threads for data loading')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--cuda', type=int, default=0,
                    help='explicitly specify to use gpu number')
# npu setting
parser.add_argument("--use_npu", action="store_true",
                    help="Use NPU to train the model")
parser.add_argument("--npu", default=0, type=int, help="NPU id to use")

# Data specifications
parser.add_argument('--dir_data', type=str, default='./dataset',
                    help='dataset directory')
parser.add_argument('--dir_train_data', type=str, default='./dataset/train/',
                    help='train dataset name')
parser.add_argument('--dir_test_data', type=str, default='./dataset/test/',
                    help='test dataset name')
parser.add_argument('--scale', type=str, default='3',
                    help='super resolution scale')

# Model specifications
parser.add_argument('--n_resblocks', type=int, default=32,
                    help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--kernel_size', type=int, default=3,
                    help='kernel size')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')

# Training specifications
parser.add_argument("--epochs", type=int, default=50,
                    help="number of epochs to train")
parser.add_argument("--batch_size", type=int, default=1,
                    help="input batch size for training")
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='learning rate decay factor for step decay')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='ADAM beta')
parser.add_argument('--step', type=int, default=10,
                    help='learning rate to the initial LR decayed by step')

args = parser.parse_args()
