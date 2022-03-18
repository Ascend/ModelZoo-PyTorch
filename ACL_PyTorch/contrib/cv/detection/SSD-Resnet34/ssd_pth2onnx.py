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
import torch
import os
from ssd300 import SSD300
import random
from argparse import ArgumentParser
from parse_config import parse_args

def pth2onnx(batch_size,input_file, output_file):
    model_options = {
        'use_nhwc' : False,
        'pad_input' : False,
        'bn_group' : 1,
    }
    ssd300_eval = SSD300(args, 81, **model_options)

    state_dict = torch.load(input_file, map_location="cpu")
    ssd300_eval.load_state_dict(state_dict['model'])

    ssd300_eval.eval()
    input_names = ["image"]
    output_names=["ploc","plabel"]
    dynamic_axes = {'image': {0: '-1'}, 'ploc': {0: '-1'}, 'plabel': {0: '-1'}}
    dummy_input = torch.randn(batch_size, 3, 300, 300)
    torch.onnx.export(ssd300_eval, dummy_input, output_file, input_names=input_names, dynamic_axes=dynamic_axes,
                      output_names=output_names, opset_version=11, verbose=False)

if __name__ == "__main__":
    args = parse_args()
    batch_size=args.bs
    input_file = args.pth_path
    output_file = args.onnx_path
    resnet_model=args.resnet34_model
    pth2onnx(batch_size,input_file, output_file)
