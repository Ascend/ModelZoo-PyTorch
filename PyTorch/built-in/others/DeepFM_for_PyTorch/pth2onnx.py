# -*- coding: utf-8 -*-

# Copyright 2022 Huawei Technologies Co., Ltd
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at# 
# 
#     http://www.apache.org/licenses/LICENSE-2.0# 
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import random
import argparse
import configparser

import numpy as np

import torch

parser = argparse.ArgumentParser(description='DeepFM for PyTorch')

parser.add_argument('--pth_path', type=str, help='pth path')
parser.add_argument('--onnx_file_path', type=str, help='out file path',default = './')
parser.add_argument('--onnx_file_name', type=str, help='name of onnx',default = 'deepfm-model.onnx')
# data config
config = configparser.ConfigParser()

def fix_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

class wrapper_model(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, xy):
        x, y = xy
        lists = []
        h ,n= x.shape
        for i in range(n):
            lists.append(x[:,i].contiguous())
        h, w ,n= y.shape
        for i in range(n):
            lists.append(y[:,:,i].contiguous())
        output = self.model(lists)
        return output

def export():
    args = parser.parse_args()
    print(args)

    Pth_root = args.pth_path
    onnx_root = os.path.join(args.onnx_file_path, args.onnx_file_name)

    model = torch.load(Pth_root)

    new_model = wrapper_model(model)
    new_model.eval()

    input_names = ["actual_input_1", "actual_input_2"]
    output_names = ["output1"]

    dummy_input_1 = torch.ones(1,26).npu()
    dummy_input_2 = torch.rand(1,1,13).npu()
    torch.onnx.export(new_model, [dummy_input_1, dummy_input_2], onnx_root, input_names=input_names,
                      output_names=output_names,
                      opset_version=11)

    print("succeed in trans pth to onnx.")

if __name__ == "__main__":
    export()

