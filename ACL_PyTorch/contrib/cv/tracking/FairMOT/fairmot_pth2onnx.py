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

import sys
import torch
import argparse

sys.path.insert(0, "./FairMOT/src")
from lib.models.model import create_model, load_model

def pth2onnx(input_file, output_file):
    arch = 'dla_34'
    heads = {'hm': 1, 'wh': 4, 'id': 128, 'reg': 2}
    head_conv = 256
    model = create_model(arch, heads, head_conv)
    load_model(model, input_file)
    model.eval()

    print('\n[INFO] Export to onnx ...')
    input_names = ["actual_input_1"]
    dynamic_axes = {'actual_input_1': {0: '-1'},
                    'hm': {0: '-1'},
                    'wh': {0: '-1'},
                    'id': {0: '-1'},
                    'reg': {0: '-1'},}
    dummy_input = torch.randn(1, 3, 608, 1088)

    torch.onnx.export(model,
                        dummy_input,
                        output_file,
                        dynamic_axes=dynamic_axes,
                        input_names=input_names,
                        output_names=['hm', 'wh', 'id','reg'], 
                        opset_version=11, 
                        verbose=True,
                        enable_onnx_checker=False)
    print('[INFO] Done!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="fairmot_dla34.pth")
    parser.add_argument("--output_file", type=str, default="fairmot.onnx")
    args = parser.parse_args()
    pth2onnx(args.input_file, args.output_file)
    