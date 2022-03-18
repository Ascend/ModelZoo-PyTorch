# Copyright 2021 Huawei Technologies Co., Ltd
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

import torch
from data.config import cfg
from pyramidbox import build_net
import numpy as np
import os
import sys

def main(onnx_path,path):
    input_names=["image"]
    output_names = ["output"]
    net = build_net('test',2)
    net.eval()
    net.load_state_dict(torch.load(path,map_location='cpu'))
    # dynamic_axes = {'image': {0: '-1'}, 'output': {0: '-1'}}
    dummy_input = torch.randn(1,3,1000,1000)
    torch.onnx.export(net,dummy_input,onnx_path,input_names = input_names,output_names=output_names,verbose=True,enable_onnx_checker=False,opset_version=9) 
    
if __name__ =="__main__":
    onnx_path = os.path.abspath(sys.argv[1])
    path = os.path.abspath(sys.argv[2])
    main(onnx_path,path)