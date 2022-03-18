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
from collections import OrderedDict
from models.shufflenetv2_wock_op_woct import shufflenet_v2_x1_0

def proc_node_module(checkpoint, AttrName):
    new_state_dict = OrderedDict()
    for k, v in checkpoint[AttrName].items():
        if (k[0:7] == "module."):
            name = k[7:]
        else:
            name = k[0:]
        new_state_dict[name] = v
    return new_state_dict


def convert(path, onnx_path, num_classes=1000):
    model = shufflenet_v2_x1_0(num_classes=num_classes)
    model.eval()

    checkpoint = torch.load(path, map_location='cpu')
    try:
        model.load_state_dict(checkpoint)
    except:
        checkpoint['state_dict'] = proc_node_module(checkpoint, 'state_dict')
        model.load_state_dict(checkpoint['state_dict'])

    input_names = ["actual_input_1"]
    output_names = ["output1"]
    dummy_input = torch.randn(16, 3, 224, 224)
    torch.onnx.export(model, dummy_input, onnx_path, input_names=input_names,
                      output_names=output_names,
                      opset_version=11)
    print('onnx export done.')


if __name__ == "__main__":
    path = "checkpoints.pth"
    onnx_path = "shufflenetv2_x1_npu_16.onnx"
    num_classes = 1000
    convert(path, onnx_path, num_classes)
