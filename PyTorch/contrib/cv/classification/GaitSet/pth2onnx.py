# -*- coding: utf-8 -*-
# Copyright (c) Soumith Chintala 2016,
# All rights reserved
#
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
# limitations under the License

from datetime import datetime
import numpy as np
import argparse

from model.initialization import initialization
from model.utils import evaluation

import os.path as osp
from apex import amp
import numpy as np
import torch
import torch.npu
import torch.onnx
import torch.nn as nn
import torch.optim as optim

from model.network import TripletLoss, SetNet
from model.utils import TripletSampler

class tmpNet(nn.Module):
    def __init__(self, module):
        super(tmpNet, self).__init__()
        self.module = module

def load(load_path, model, restore_iter):
    loaded = torch.load(load_path, map_location=torch.device('cpu'))
    model.load_state_dict(loaded)

def convert(input_path, output_path, restore_iter):
    align_size = 100
    hidden_dim = 256
    
    # load model checkpoint of iteration args.iter
    print(f'Init the model of iteration {restore_iter}...')
    encoder = SetNet(hidden_dim).float()
    encoder = encoder.npu()
    encoder = tmpNet(encoder)
    
    print(f'Loading the model of iteration {restore_iter}...')
    load(input_path, encoder, restore_iter)
    
    encoder.eval()
    input_names = ["image_seq"]
    output_names = ["feature"]
    dummy_input = torch.randn((1, align_size, 64, 44), requires_grad=False).npu()
    
    print('Exporting model to onnx...')
    torch.onnx.export(encoder.module, dummy_input, output_path, input_names = input_names, output_names = output_names, opset_version=11, verbose=False)


if __name__ == "__main__":
    from config import conf_8p
    conf_model = conf_8p['model']
    
    frame_num = conf_model['frame_num']
    restore_iter = conf_model['restore_iter']
    path = f"work/checkpoint/GaitSet/GaitSet_CASIA-B_73_False_256_0.2_128_full_{frame_num}-{restore_iter}-encoder.ptm"
    onnx_path = "gaitset_submit.onnx"
    convert(path, onnx_path, restore_iter)
