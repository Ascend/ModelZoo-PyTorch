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

#coding=utf-8
#torch.__version >= 1.3.0
import sys
sys.path.append("..")

import torch.onnx
from models.factory import build_net
import argparse


parser = argparse.ArgumentParser(description="trans pth to onnx usage")
parser.add_argument( '--model_path', type=str, default='../dsfd.pth', help='Default ph model location(default: %(default)s)')
args = parser.parse_args()


#Function to Convert to ONNX
def Convert_ONNX(model):
    print("enter Convert_ONNX")

    # set the model to inference mode
    model.eval()

    # 构建输入信息和输出信息
    input_names = ["image"]
    output_names = ["modelOutput1", "modelOutput2", "modelOutput3", "modelOutput4", "modelOutput5", "modelOutput6"]
    #dynamic_axes = {'image': {0: '4'}, 'modelOutput': {0: '-1'}}
    dynamic_axes = {'image': {0: '4'}, 'modelOutput1': {0: '4'}, 'modelOutput2': {0: '4'}, 'modelOutput3': {0: '4'},
                    'modelOutput4': {0: '4'}, 'modelOutput5': {0: '4'},'modelOutput6': {0: '4'}}
    #dynamic_axes = {'image': {0: '4'}, 'modelOutput': {0: '4'}}
    dummy_input = torch.randn(4, 3, 224, 224)

    # 开始转换
    torch.onnx.export(model,
                      dummy_input,
                      "dsfd.onnx",
                      input_names=input_names,
                      dynamic_axes=dynamic_axes,
                      output_names=output_names,
                      opset_version=11,
                      verbose=True)
    print("*************Convert to ONNX model file SUCCESS!*************")


if __name__ == '__main__':

    model = build_net('train', 2, 'resnet152')
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
    Convert_ONNX(model)





