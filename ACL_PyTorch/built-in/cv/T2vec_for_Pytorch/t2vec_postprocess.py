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

import h5py
import torch
import torch.nn as nn
import numpy as np
from data_utils import DataOrderScaner

def parse_args():
    parser = argparse.ArgumentParser(description="Configs")

    parser.add_argument("-result_dir", default="./result",
        help="Path to training and validating data")

    parser.add_argument("-data", default="./data",
        help="Path to training and validating data")
    
    parser.add_argument("-num_layers", type=int, default=3,
        help="Number of layers in the RNN cell")

    parser.add_argument("-prefix", default="exp1", help="Prefix of trjfile")

    parser.add_argument("-t2vec_batch", type=int, default=256,
        help="The maximum number of trajs we encode each time in t2vecs")

    args = parser.parse_args()
    return args


def postprocess(args):
    scaner = DataOrderScaner(os.path.join(args.data, "{}-trj.t".format(args.prefix)), args.t2vec_batch)
    scaner.load()
    vecs = []
    i = 0
    while True:
        if i % 100 == 0:
            print("{}: Encoding {} trjs...".format(i, args.t2vec_batch))
        i = i + 1
        src, lengths, invp = scaner.getbatch()
        if src is None: 
            break
        h = np.load(os.path.join(args.result_dir, f"{i}_0.npy"))
        h = torch.from_numpy(h)
        h = h.transpose(0, 1).contiguous()
        vecs.append(h[invp].cpu().data)
    vecs = torch.cat(vecs)
    vecs = vecs.transpose(0, 1).contiguous()
    path = os.path.join(args.data, "{}-trj.h5".format(args.prefix))
    with h5py.File(path, "w") as f:
        for i in range(args.num_layers):
            f["layer"+str(i+1)] = vecs[i].squeeze(0).numpy()


if __name__=="__main__":
    opts = parse_args()
    postprocess(opts)