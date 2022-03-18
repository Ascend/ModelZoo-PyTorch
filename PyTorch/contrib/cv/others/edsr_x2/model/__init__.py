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
import os
from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.parallel as P
import torch.utils.model_zoo


class Model(nn.Module):
    def __init__(self, args, ckp):
        super(Model, self).__init__()
        print("Making model...")

        self.scale = args.scale
        self.idx_scale = 0
        self.input_large = args.model == "VDSR"
        self.cpu = args.cpu
        self.device = torch.device("cpu" if args.cpu else "cuda")
        if args.use_npu:
            self.device = args.device
        self.n_GPUs = args.n_GPUs
        self.save_models = args.save_models

        module = import_module("model." + args.model.lower())
        self.model = module.make_model(args).to(self.device)

        self.load(pre_train=args.pre_train, cpu=args.cpu)
        print(self.model, file=ckp.log_file)

    def forward(self, x, idx_scale):
        self.idx_scale = idx_scale
        if hasattr(self.model, "set_scale"):
            self.model.set_scale(idx_scale)

        if self.training:
            if self.n_GPUs > 1:
                return P.data_parallel(self.model, x, range(self.n_GPUs))
            else:
                return self.model(x)
        else:
            forward_function = self.model.forward
            return forward_function(x)

    def save(self, apath, epoch, is_best=False):
        save_dirs = [os.path.join(apath, "model_latest.pt")]

        if is_best:
            save_dirs.append(os.path.join(apath, "model_best.pt"))
        if self.save_models:
            save_dirs.append(os.path.join(apath, "model_{}.pt".format(epoch)))

        for s in save_dirs:
            torch.save(self.model.state_dict(), s)

    def load(self, pre_train="", cpu=False):
        load_from = None
        kwargs = {}
        if cpu:
            kwargs = {"map_location": lambda storage, loc: storage}
        if pre_train:
            load_from = torch.load(os.path.join(pre_train), **kwargs)
            isFind = False
            for k, _ in load_from.items():
                if k == "model":
                    isFind = True
            if isFind:
                new_load_from = load_from["model"]
                load_from = new_load_from
        if load_from:
            self.model.load_state_dict(load_from, strict=False)
