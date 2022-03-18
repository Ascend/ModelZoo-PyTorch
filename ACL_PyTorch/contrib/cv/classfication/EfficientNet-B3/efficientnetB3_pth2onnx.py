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

import sys
import torch
import torch.onnx
from collections import OrderedDict
sys.path.append(r"./pycls")
from pycls.models.effnet import EffNet
import pycls.core.config as config
from pycls.core.config import cfg

def proc_node_module(checkpoint, attr_name):
    new_model_state = OrderedDict()
    for k, v in checkpoint[attr_name].items():
        if(k[0: 7] == "module."):
            name = k[7:]
        else:
            name = k[0:]
        new_model_state[name] = v
    return new_model_state
    
def convert(input_file_, yaml_file_, output_file_):
    config.load_cfg(yaml_file_)
    cfg.freeze()

    model = EffNet()
    checkpoint = torch.load(input_file_, map_location='cpu')
    checkpoint['model_state'] = proc_node_module(checkpoint, 'model_state')
    model.load_state_dict(checkpoint["model_state"]) 
    model.eval()    
   
    input_names = ["image"]
    output_names = ["class"]
    dynamic_axes = {'image': {0: '-1'}, 'class': {0: '-1'}}
    dummy_input = torch.randn(16, 3, 300, 300)
    torch.onnx.export(model, dummy_input, output_file_, input_names=input_names, dynamic_axes=dynamic_axes,
                      output_names=output_names, opset_version=11, verbose=True) 
   
if __name__ == "__main__":
    input_file = sys.argv[1]
    yaml_file= sys.argv[2]
    output_file = sys.argv[3]
    convert(input_file, yaml_file, output_file)
