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

# coding=utf-8
import sys
sys.path.append('./RCF-pytorch')
import os
import argparse
import torch
from models import RCF


# change pth to onnx
def rcf_pth2onnx(args):
    """[change pth to onnx]

    Args:
        args ([argparse]): [change pth to onnx parameters]
    """
    assert torch.cuda.is_available(), print('The model must be loaded on GPU')
    device = torch.device("cuda:0")
    model = RCF() # RCF model
    model.to(device)
    checkpoint = torch.load(args.pth_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    input_names = ['image']
    output_names = ['output1', 'output2', 'output3', 'output4', 'output5', 'output6']
    dummy_input = torch.randn(1, 3, 321, 481).to(device)
    dynamic_axes = {"image": {0: "-1", 2: "-1", 3: "-1"},
                    "output1": {0: "-1", 2: "-1", 3: "-1"},
                    "output2": {0: "-1", 2: "-1", 3: "-1"},
                    "output3": {0: "-1", 2: "-1", 3: "-1"},
                    "output4": {0: "-1", 2: "-1", 3: "-1"},
                    "output5": {0: "-1", 2: "-1", 3: "-1"},
                    "output6": {0: "-1", 2: "-1", 3: "-1"},}
    torch.onnx.export(model, dummy_input, args.onnx_name, input_names=input_names,
                      output_names=output_names, opset_version=11, verbose=True,
                      dynamic_axes=dynamic_axes)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='rcf pth to onnx') # change model parameters
    parser.add_argument('-p', '--pth_path', default='RCFcheckpoint_epoch12.pth',
                        type=str, help='pth model path')
    parser.add_argument('-o', '--onnx_name', default='rcf.onnx',
                        type=str, help='onnx model path')
    args = parser.parse_args()
    rcf_pth2onnx(args)