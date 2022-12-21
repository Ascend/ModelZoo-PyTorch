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


import sys

import torch
import torch.onnx

sys.path.append(r"./pytorch-ssd")
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd


def pytorch2onnx(ckpt_path, out_path):
    num_classes = 21
    net = create_mobilenetv1_ssd(num_classes, is_test=True)
    print("begin to load model")
    net.load(ckpt_path)
    net.eval()

    input_names = ["image"]
    output_names = ['scores', 'boxes']
    dynamic_axes = {
        'image': {0: '-1'}, 
        'scores': {0: '-1'}, 
        'boxes': {0: '-1'}
    }
    dummy_input = torch.randn(16, 3, 300, 300)
    print("begin to export")
    torch.onnx.export(net, dummy_input, out_path, 
                      input_names=input_names,
                      output_names=output_names, 
                      dynamic_axes=dynamic_axes, 
                      opset_version=11, 
                      verbose=False)
    print("end export")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser('Pytorch model convert to ONNX')
    parser.add_argument('--ckpt', default=None, 
                        help='input checkpoint file path')
    parser.add_argument('--onnx', default='out.onnx', 
                        help='output onnx file path')
    args = parser.parse_args()

    pytorch2onnx(args.ckpt, args.onnx)
