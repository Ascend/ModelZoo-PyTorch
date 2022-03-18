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
import sys
import os
from GloRe.network.resnet50_3d_gcn_x5 import RESNET50_3D_GCN_X5
def pth2onnx(input_file, output_file):
    net = RESNET50_3D_GCN_X5(num_classes=101, pretrained=False)
    state_dict = torch.load(input_file,map_location='cpu')
    net.load_state_dict(state_dict['state_dict'])
    net.eval()
    input_names = ["image"]
    output_names = ["class"]
    dynamic_axes = {'image': {0: '-1'}, 'class': {0: '-1'}}
    dummy_input = torch.randn(1, 3, 8, 224, 224)
    torch.onnx.export(net, dummy_input, output_file, input_names=input_names, dynamic_axes=dynamic_axes,
                      output_names=output_names, opset_version=11, verbose=True)


if __name__ == '__main__':
    args = sys.argv
    pth2onnx(args[1], args[2])
