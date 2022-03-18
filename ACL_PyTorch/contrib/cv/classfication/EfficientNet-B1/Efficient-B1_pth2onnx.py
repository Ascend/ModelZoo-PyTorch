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
import argparse
import sys
import torch
from collections import OrderedDict
sys.path.append('./pycls')
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

def main(input_file, yaml_file, output_file):
    config.load_cfg(yaml_file)
    cfg.freeze()
    model = EffNet()
    checkpoint = torch.load(input_file, map_location='cpu')
    checkpoint['model_state'] = proc_node_module(checkpoint, 'model_state')
    model.load_state_dict(checkpoint["model_state"]) 
    model.eval()  
    input_names = ["image"]
    dynamic_axes = {'image': {0: '-1'}, 'class': {0: '-1'}}
    dummy_input = torch.randn(1, 3, 240, 240)
    torch.onnx.export(model, dummy_input, output_file, input_names=input_names, dynamic_axes=dynamic_axes, opset_version=11, verbose=True)


if __name__ == '__main__':
    input_file = sys.argv[1]
    yaml_file= sys.argv[2]
    output_file = sys.argv[3]
    main(input_file, yaml_file, output_file)
