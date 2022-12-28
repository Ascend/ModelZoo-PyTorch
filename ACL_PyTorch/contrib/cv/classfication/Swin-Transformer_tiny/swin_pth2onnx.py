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

from collections import OrderedDict
import sys
import torch

sys.path.append('./Swin-Transformer')
from models import build_model
from main import parse_option


def proc_nodes_module(checkpoint, AttrName):
    new_state_dict = OrderedDict()
    for k, v in checkpoint[AttrName].items():
        if "module." in k:
            name = k.replace("module.", "")
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict


def model_transfer(config):
    batch_size = config.DATA.BATCH_SIZE
    onnx_model = build_model(config)
    ckpt = torch.load(config.MODEL.RESUME, map_location='cpu')
    ckpt['model'] = proc_nodes_module(ckpt, 'model')
    onnx_model.load_state_dict(ckpt['model'])
    onnx_model.cpu()
    input_names = ["image"]
    output_names = ["class"]
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    fp16 = False
    if fp16:
        onnx_path = 'models/onnx/swin_tiny_bs' + str(batch_size) + '_fp16.onnx'
    else:
        onnx_path = 'models/onnx/swin_tiny_bs' + str(batch_size) + '.onnx'
    torch.onnx.export(onnx_model, dummy_input, onnx_path, input_names=input_names,
                      output_names=output_names, opset_version=11, verbose=True)
    print("model saved successful at ", onnx_path)


if __name__ == '__main__':
    _, main_config = parse_option()
    model_transfer(main_config)
