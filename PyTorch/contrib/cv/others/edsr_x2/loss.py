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

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.modules.loss._Loss):
    def __init__(self, args, ckp):
        super(Loss, self).__init__()
        print("Preparing loss function:")

        self.n_GPUs = args.n_GPUs
        self.loss = []
        self.loss_module = nn.ModuleList()
        for loss in args.loss.split("+"):
            weight, loss_type = loss.split("*")
            if loss_type == "MSE":
                loss_function = nn.MSELoss()
            elif loss_type == "L1":
                loss_function = nn.L1Loss()
            elif loss_type.find("VGG") >= 0:
                module = import_module("loss.vgg")
                loss_function = getattr(module, "VGG")(
                    loss_type[3:], rgb_range=args.rgb_range
                )
            elif loss_type.find("GAN") >= 0:
                module = import_module("loss.adversarial")
                loss_function = getattr(module, "Adversarial")(args, loss_type)

            self.loss.append(
                {"type": loss_type, "weight": float(
                    weight), "function": loss_function}
            )
            if loss_type.find("GAN") >= 0:
                self.loss.append(
                    {"type": "DIS", "weight": 1, "function": None})

        if len(self.loss) > 1:
            self.loss.append({"type": "Total", "weight": 0, "function": None})

        for l in self.loss:
            if l["function"] is not None:
                print("{:.3f} * {}".format(l["weight"], l["type"]))
                self.loss_module.append(l["function"])

        self.log = torch.Tensor()

        device = torch.device("cpu" if args.cpu else "cuda")
        if args.use_npu:
            device = args.device
        self.loss_module.to(device)
        if not args.cpu and args.n_GPUs > 1:
            self.loss_module = nn.DataParallel(
                self.loss_module, range(args.n_GPUs))

        if args.load != "":
            self.load(ckp.dir, cpu=args.cpu)

    def forward(self, sr, hr):
        losses = []
        for i, l in enumerate(self.loss):
            if l["function"] is not None:
                loss = l["function"](sr, hr)
                effective_loss = l["weight"] * loss
                losses.append(effective_loss)
                self.log[-1, i] += effective_loss.item()
            elif l["type"] == "DIS":
                self.log[-1, i] += self.loss[i - 1]["function"].loss

        loss_sum = sum(losses)
        if len(self.loss) > 1:
            self.log[-1, -1] += loss_sum.item()

        return loss_sum

    def step(self):
        for l in self.get_loss_module():
            if hasattr(l, "scheduler"):
                l.scheduler.step()

    def start_log(self):
        self.log = torch.cat((self.log, torch.zeros(1, len(self.loss))))

    def end_log(self, n_batches):
        self.log[-1].div_(n_batches)

    def display_loss(self, batch):
        n_samples = batch + 1
        log = []
        for l, c in zip(self.loss, self.log[-1]):
            log.append("[{}: {:.4f}]".format(l["type"], c / n_samples))

        return "".join(log)

    def get_loss_module(self):
        if self.n_GPUs == 1:
            return self.loss_module
        else:
            return self.loss_module.module

    def save(self, apath):
        torch.save(self.state_dict(), os.path.join(apath, "loss.pt"))
        torch.save(self.log, os.path.join(apath, "loss_log.pt"))

    def load(self, apath, cpu=False):
        if cpu:
            kwargs = {"map_location": lambda storage, loc: storage}
        else:
            kwargs = {}

        self.load_state_dict(torch.load(
            os.path.join(apath, "loss.pt"), **kwargs))
        self.log = torch.load(os.path.join(apath, "loss_log.pt"))
        for l in self.get_loss_module():
            if hasattr(l, "scheduler"):
                for _ in range(len(self.log)):
                    l.scheduler.step()
