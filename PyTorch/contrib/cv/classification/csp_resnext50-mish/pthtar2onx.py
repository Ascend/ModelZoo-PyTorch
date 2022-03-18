# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



import torch
import torch.onnx
from timm.models import create_model
from collections import OrderedDict

import argparse

parser = argparse.ArgumentParser(description='mobilenetv3_large_100')
parser.add_argument('--model-path', default='', type=str, metavar='PATH',
                    help='model path')

def createModel():
     model = create_model(
        'cspresnext50',
        num_classes=1000)
     print(model)
     return model
def proc_node_module(checkpoint, AttrName):
    new_state_dict = OrderedDict()
    for k, v in checkpoint[AttrName].items():
        if(k[0:7] == "module."):
            name = k[7:]
        else:
            name = k[0:]
        new_state_dict[name] = v
    return new_state_dict


def convert():

    args = parser.parse_args()
    model_path = args.model_path
    checkpoint = torch.load(model_path, map_location='cpu')
    #print(checkpoint.keys())
    model = createModel()
    checkpoint['state_dict'] = proc_node_module(checkpoint, 'state_dict')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    #print(model)

    input_names = ["actual_input_1"]
    output_names = ["output1"]
    dummy_input = torch.randn(32, 3, 224, 224)
    torch.onnx.export(model, dummy_input, "csp_resnext50-mish_npu_32.onnx", input_names=input_names, output_names=output_names, opset_version=11)


if __name__ == "__main__":
    convert()
