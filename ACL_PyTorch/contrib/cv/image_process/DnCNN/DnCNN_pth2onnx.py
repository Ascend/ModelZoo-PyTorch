# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import torch
import torch.onnx
import torch.nn as nn
import sys

from collections import OrderedDict

class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, \
                      kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, \
                          kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, \
                      kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
    
        out = self.dncnn(x)
        return out


def proc_nodes_module(checkpoint):

    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        if(k[0:7] == "module."):
            name = k[7:]
        else:
            name = k[0:]
        new_state_dict[name]=v
    return new_state_dict


def convert(pth_file, onnx_file):

    pretrained_net = torch.load(pth_file, map_location='cpu')
    pretrained_net['state_dict'] = proc_nodes_module(pretrained_net)

    model = DnCNN(channels=1, num_of_layers=17)
    model.load_state_dict(pretrained_net['state_dict'])
    model.eval()
    input_names = ["actual_input_1"]
    dummy_input = torch.randn(1, 1, 481, 481)
    #torch.onnx.export(model, dummy_input, onnx_file, input_names = input_names, opset_version=11, verbose=True)
    dynamic_axes = {'actual_input_1': {0: '-1'}}
    torch.onnx.export(model, dummy_input, onnx_file, dynamic_axes=dynamic_axes, \
                      input_names=input_names, opset_version=11)

if __name__ == "__main__":
    
    pth_file = sys.argv[1]
    onnx_file = sys.argv[2]

    convert(pth_file, onnx_file)
