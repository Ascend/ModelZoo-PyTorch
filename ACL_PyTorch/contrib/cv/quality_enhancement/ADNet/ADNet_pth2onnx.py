
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
from models import ADNet
import torch
import torch.onnx
import sys
from collections import OrderedDict
def proc_nodes_module(checkpoint):
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        if "module." in k:
            name = k.replace("module.", "")
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict

def pth2onnx(path, output_file1):
    net = ADNet(channels=1, num_of_layers=17)
    model = net #model = nn.DataParallel(net, device_ids=device_ids).cuda()
    checkpoint = torch.load(path, map_location='cpu')
    checkpoint = proc_nodes_module(checkpoint)
    model.load_state_dict(checkpoint)
    model.eval()
    input_names = ["image"]
    output_names = ["output1"]
    dynamic_axes = {'image': {0: '-1'}, 'output1': {0: '-1'}}
    dummy_input1 = torch.randn(1, 1, 321, 481)
    torch.onnx.export(model, dummy_input1, output_file1, input_names = input_names, dynamic_axes = dynamic_axes,output_names = output_names, opset_version=11, verbose=True)
    print("ADNET onnx has transformed successfully and this model supports dynamic axes")
    print('onnx export done.')

if __name__ == "__main__":
    path = sys.argv[1]
    onnx_path = sys.argv[2]
    pth2onnx(path, onnx_path)