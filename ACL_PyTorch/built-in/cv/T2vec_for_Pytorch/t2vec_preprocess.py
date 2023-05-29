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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_utils import DataOrderScaner


def parse_args():
    parser = argparse.ArgumentParser(description="Configs")

    parser.add_argument("-save_dir", default="./prep_data",
        help="Path to training and validating data")

    parser.add_argument("-data", default="./data",
        help="Path to training and validating data")

    parser.add_argument("-prefix", default="exp1", help="Prefix of trjfile")

    parser.add_argument("-t2vec_batch", type=int, default=256,
        help="The maximum number of trajs we encode each time in t2vecs")

    args = parser.parse_args()
    return args


def cal_upper_bound(num):
    if num > 100:
        raise ValueError("Length of sequence is out of 100 which is not support") 
    else:
        upper_bound = (num // 10 + 1) * 10
        return upper_bound


def preprocess(args):
    scaner = DataOrderScaner(os.path.join(args.data, "{}-trj.t".format(args.prefix)), args.t2vec_batch)
    scaner.load()
    i = 0
    while True:
        if i % 100 == 0:
            print("{}: Encoding {} trjs...".format(i, args.t2vec_batch))
        i = i + 1
        src, lengths, invp = scaner.getbatch()
        if src is None: 
            break

        # pad to t2vec_batch
        if src.shape[-1] != args.t2vec_batch:
            p1d = (0, args.t2vec_batch - src.shape[-1])
            src = F.pad(src, p1d, "constant", 0)
            lengths = F.pad(lengths, p1d, "constant", 1)
        
        # pad to supported seq_len
        seq_len = src.shape[0]
        upper_bound = cal_upper_bound(seq_len)
        p1d = (0, 0, 0, upper_bound - seq_len)
        src = F.pad(src, p1d, "constant", 0)
 
        # save inputs data of model
        for inp in ["src", "lengths", "invp"]:
            save_path = os.path.join(args.save_dir, f"{inp}/{i}.npy")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, eval(f"{inp}.numpy()"))


if __name__=="__main__":
    opts = parse_args()
    preprocess(opts)
