# Copyright 2022 Huawei Technologies Co., Ltd
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
import sys

def pth2onnx(input_file, output_file):
    model = torch.load(input_file, map_location = torch.device('cpu'))  # map_location为转换设备
    model.eval()
    input_names = ["image"]
    output_names = ["class"]
    dynamic_axes = {'image': {0: '-1'}, 'class': {0: '-1'}} 
    dummy_input = torch.randn(1, 3, 128, 128)  # 模型输入的shape
    torch.onnx.export(model.module, dummy_input, output_file,
                      input_names=input_names,dynamic_axes = dynamic_axes,
                      output_names=output_names, opset_version=13, verbose=False) # opset是可视化用的,verbose输出进度条显示

def main():
    input_file = sys.argv[1] 
    output_file = sys.argv[2]
    pth2onnx(input_file, output_file)

if __name__ == '__main__':
    main()
