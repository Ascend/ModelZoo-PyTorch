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
import torch.onnx
from collections import OrderedDict
from pycls.models.effnet import EffNet
from pycls.core.net import unwrap_model
from iopath.common.file_io import g_pathmgr
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
    
def convert():
    """Sets up a model for training or testing and log the results."""
    loc = 'cpu'
    with g_pathmgr.open("result/model.pyth", "rb") as f:
        checkpoint = torch.load(f, map_location=loc)
    print(checkpoint.keys())
    config.merge_from_file('configs/dds_baselines/effnet/EN-B3_dds_8npu.yaml')
    cfg.freeze()
    model = EffNet()
    print(model)
    checkpoint['model_state'] = proc_node_module(checkpoint, 'model_state')
    model.load_state_dict(checkpoint["model_state"], False) 
    model.eval()    
   
    input_names = ["actual_input_1"]
    output_names = ["output1"]
    dummy_input = torch.randn(32, 3, 456, 456).to(loc)
    torch.onnx.export(model, dummy_input, "efficientnetB3_npu.onnx", input_names = input_names, output_names = output_names, opset_version=11)   
   
if __name__ == "__main__":
    convert()
