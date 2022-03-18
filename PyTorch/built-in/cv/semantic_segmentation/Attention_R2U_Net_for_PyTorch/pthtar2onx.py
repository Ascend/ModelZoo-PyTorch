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


import torch
import torch.onnx

from collections import OrderedDict
from network import R2AttU_Net


def proc_nodes_module(checkpoint):
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        if (k[0:7] == "module."):
            name = k[7:]
        else:
            name = k[0:]
        new_state_dict[name] = v
    return new_state_dict


def convert(pth_file_path, onnx_file_path):
    checkpoint = torch.load(pth_file_path, map_location='cpu')
    
    model_checkpoint = checkpoint["model"]
    if list(model_checkpoint.keys())[0].startswith("module."):
        model_checkpoint = proc_nodes_module(model_checkpoint)

    model = R2AttU_Net(img_ch=3,output_ch=1,t=2)
    model.load_state_dict(model_checkpoint)
    model.eval()
    print(model)

    input_names = ["actual_input_1"]
    output_names = ["output1"]
    dummy_input = torch.randn(4, 3, 224, 224)
    torch.onnx.export(model, dummy_input, onnx_file_path, input_names=input_names, output_names=output_names,
                      opset_version=11)


if __name__ == "__main__":
    src_file_path = "checkpoint.pkl"
    dst_file_path = "R2AttU_Net.onnx"
    convert(src_file_path, dst_file_path)