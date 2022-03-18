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
from efficientnet_pytorch.model import EfficientNet


def proc_node_module(checkpoint, attr_name):
    """
    modify state_dict
    :param checkpoint: loaded model file
    :param attr_name: key state_dict
    :return: new state_dict
    """
    new_state_dict = OrderedDict()
    for k, v in checkpoint[attr_name].items():
        if k[0:7] == "module.":
            name = k[7:]
        else:
            name = k[0:]
        new_state_dict[name] = v
    return new_state_dict


def convert(pth_file_path, onnx_file_path, class_nums):
    """
    convert pth file to onnx file and output onnx file
    """
    checkpoint = torch.load(pth_file_path, map_location='cpu')
    checkpoint['state_dict'] = proc_node_module(checkpoint, 'state_dict')
    model = EfficientNet.from_name("efficientnet-b0", num_classes=class_nums)
    model.set_swish(memory_efficient=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    input_names = ["actual_input_1"]
    output_names = ["output1"]
    dummy_input = torch.randn(16, 3, 224, 224)
    torch.onnx.export(model, dummy_input, onnx_file_path,
                      input_names=input_names, output_names=output_names,
                      opset_version=11)


if __name__ == "__main__":
    src_file_path = "./checkpoint.pth"
    dst_file_path = "efficientnet_npu_16.onnx"
    class_num = 1000
    convert(src_file_path, dst_file_path, class_num)