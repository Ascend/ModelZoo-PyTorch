#!/usr/bin/env python3
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://spdx.org/licenses/BSD-3-Clause.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import numpy as np
import os.path as osp
import os
import argparse

from network import MGN
from data import Data
from opt import opt


def build_model():
    # Create model
    model = MGN()
    model.load_state_dict(torch.load(opt.weight))
    if opt.npu:
        model = model.to("npu:0")
    model.eval()
    return model


def get_raw_data(data):
    inputs, targets = next(iter(data.query_loader))
    return inputs, targets

def extract_one_batch_feature(model, inputs):
    if opt.npu:
        inputs = inputs.npu()
    outputs = model(inputs)
    f1 = outputs[0].data.cpu()

    # flip
    inputs = inputs.index_select(3, torch.arange(inputs.size(3) - 1, -1, -1))
    if opt.npu:
        inputs = inputs.npu()
    outputs = model(inputs)
    f2 = outputs[0].data.cpu()
    ff = f1 + f2

    fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
    ff = ff.div(fnorm.expand_as(ff))
    return ff


if __name__ == '__main__':
    data = Data()
    inputs, targets = get_raw_data(data)
    model = build_model()

    qf = extract_one_batch_feature(model, inputs).numpy()
    print(qf)
