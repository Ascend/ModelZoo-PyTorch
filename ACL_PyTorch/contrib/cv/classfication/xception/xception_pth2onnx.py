# Copyright 2023 Huawei Technologies Co., Ltd
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

import sys
import argparse
import torch
sys.path.append(r"./Xception-PyTorch")
import torch.onnx
from xception import xception

def pth2onnx(input_file, output_file):
    model = xception(pretrained=False)
    checkpoint = torch.load(input_file, map_location=None)
    model.load_state_dict(checkpoint)

    model.eval()
    
    
    input_names = ["image"]
    output_names = ["class"]
    dynamic_axes = {'image': {0: '-1'}, 'class': {0: '-1'}}
    dummy_input = torch.randn(1, 3, 299, 299)
    torch.onnx.export(model, dummy_input, output_file, input_names = input_names, dynamic_axes = dynamic_axes, output_names = output_names, verbose=True, opset_version=11)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocess of MaskRCNN PyTorch model')
    parser.add_argument("--input_file", default="./coco2017/", help='image of dataset')
    parser.add_argument("--output_file", default="./coco2017_bin/", help='Preprocessed image buffer')
    flags = parser.parse_args()    

    pth2onnx(flags.input_file, flags.output_file)

