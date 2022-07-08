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
import os
import sys
import torch
import onnx
from pretrainedmodels.models.senet import se_resnext50_32x4d

def pth2onnx(input_file, output_file):
    model = se_resnext50_32x4d(num_classes=1000, pretrained='imagenet')
    model.load_state_dict(torch.load(input_file))
    model.eval()
    input_names = ["image"]
    output_names = ["class"]
    dynamic_axes = {'image': {0: '-1'}, 'class': {0: '-1'}}
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy_input, output_file, input_names = input_names, dynamic_axes = dynamic_axes, output_names = output_names, opset_version=11)
    
if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise Exception("usage: python seresnext50_pth2onnx.py <input_file> <output_file>")
    input_file = sys.argv[1] 
    output_file = sys.argv[2]
    pth2onnx(input_file, output_file)
