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
import torchvision.models as models
from collections import OrderedDict


def proc_node_module(checkpoint, attr_name):
    new_state_dict = OrderedDict()
    for k, v in checkpoint[attr_name].items():
        if(k[0:7] == "module."):
            name = k[7:]
        else:
            name = k[0:]
        new_state_dict[name] = v
    return new_state_dict


def convert(bs, ckpt_path="./model_lincls_best.pth.tar"):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    checkpoint['state_dict'] = proc_node_module(checkpoint, 'state_dict')
    model = models.__dict__['resnet50']()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    input_names = ["actual_input_1"]
    output_names = ["output1"]
    dummy_input = torch.randn(bs, 3, 224, 224)
    torch.onnx.export(model, dummy_input, "moco-v2-bs" + str(bs) + ".onnx", input_names=input_names, output_names=output_names, opset_version=11)


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:
        convert(int(sys.argv[1]), sys.argv[2])
    elif len(sys.argv) == 2:
         convert(int(sys.argv[1]))