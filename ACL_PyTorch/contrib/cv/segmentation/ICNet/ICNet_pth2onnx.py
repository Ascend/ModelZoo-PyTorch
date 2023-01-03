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
import torch.onnx
import sys
sys.path.append('./ICNet-pytorch')
from models import ICNet

def convert(pth_file, onnx_file):
    
    model = ICNet(nclass=19, backbone='resnet50', pretrained_base=False, train_mode=False)
    print(model)
    pretrained_net = torch.load(pth_file, map_location='cpu')
    model.load_state_dict(pretrained_net)
    model.eval()
    input_names = ["actual_input_1"]
    dummy_input = torch.randn(1, 3, 1024, 2048)
    dynamic_axes = {'actual_input_1': {0: '-1'}}
    torch.onnx.export(model, dummy_input, onnx_file, dynamic_axes=dynamic_axes, input_names=input_names, opset_version=11)
    # torch.onnx.export(model, dummy_input, onnx_file, input_names=input_names, opset_version=11)

if __name__ == "__main__":

    pth_file = sys.argv[1]
    onnx_file = sys.argv[2]

    convert(pth_file, onnx_file)
