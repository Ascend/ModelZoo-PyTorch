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
# limitations under the License

import sys
sys.path.append("./LResNet")

import torch
from model import Backbone



def pth2onnx(input_file, output_file, batch_size):
    device = torch.device('cpu')

    # 加载模型权重并设置模式为eval
    model = Backbone(num_layers=100, drop_ratio=1, mode='ir_se').to(device)
    ckpt = torch.load(input_file, map_location='cpu')
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # 构建输入信息和输出信息
    input_names = ["image"]
    output_names = ["features"]
    dynamic_axes = {'image': {0: f'{batch_size}'}, 'features': {0: f'{batch_size}'}}
    dummy_input = torch.randn(batch_size, 3, 112, 112)

    # 开始转换
    torch.onnx.export(model,
                      dummy_input,
                      output_file,
                      input_names=input_names,
                      dynamic_axes=dynamic_axes,
                      output_names=output_names,
                      opset_version=11,
                      verbose=True)
    print("*************Convert to ONNX model file SUCCESS!*************")


if __name__ == '__main__':
    pth2onnx(sys.argv[1], sys.argv[2], int(sys.argv[3]))