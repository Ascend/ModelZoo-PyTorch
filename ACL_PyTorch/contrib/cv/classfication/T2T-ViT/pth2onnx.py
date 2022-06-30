# Copyright 2020 Huawei Technologies Co., Ltd
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

from models.t2t_vit import *
from utils import load_for_transfer_learning
import torch

def pth2onnx(input_file, output_file):
    # create model
    model = t2t_vit_14()

    # load the pretrained weights
    load_for_transfer_learning(model, input_file, use_ema=True, strict=False, num_classes=1000)
    # 调整模型为eval mode
    model.eval()
    # 输入节点名
    input_names = ["image"]
    # 输出节点名
    output_names = ["class"]
    dynamic_axes = {'image': {0: '-1'}, 'class': {0: '-1'}}
    dummy_input = torch.randn(1, 3, 224, 224)
    # verbose=True，支持打印onnx节点和对应的PyTorch代码行
    torch.onnx.export(model, dummy_input, output_file, input_names = input_names, dynamic_axes = dynamic_axes, output_names = output_names, opset_version=11, verbose=True, enable_onnx_checker=False)

if __name__ == '__main__':
    pth2onnx('81.5_T2T_ViT_14.pth.tar','T2T_ViT_14.onnx')