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
import torch.onnx
import sys
sys.path.append(r"./pytorch-ssd")
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd


def pth2onx(model_path, out_path):
    num_classes = 21
    net = create_mobilenetv1_ssd(num_classes, is_test=True)
    print("begin to load model")
    net.load(model_path)
    net.eval()

    input_names = ["image"]
    dynamic_axes = {'image': {0: '-1'}, 'scores':{0: '-1'}, 'boxes': {0: '-1'}}
    output_names = ['scores', 'boxes']
    dummy_input = torch.randn(16, 3, 300, 300)
    print("begin to export")
    torch.onnx.export(net, dummy_input, out_path, input_names=input_names,
                      dynamic_axes=dynamic_axes, output_names=output_names, opset_version=11, verbose=True)
    print("end export")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python SSD_MobileNet_pth2onnx.py  <model path> <out path>')
        sys.exit(0)

    model_path = sys.argv[1]
    out_path = sys.argv[2]
    pth2onx(model_path, out_path)
