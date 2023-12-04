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
import ssl
import torchvision.models as models
import argparse


def convert(args):
    model = models.vgg16(pretrained=False)
    ckpt = torch.load(args.pth_path, map_location=torch.device('cpu'))
    model.load_state_dict(ckpt)
    model.eval()
    input_names = ["actual_input_1"]
    output_names = ["output1"]
    dummy_input = torch.randn(16, 3, 224, 224)
    dynamic_axes = {'actual_input_1': {0: '-1'}, 'output1': {0: '-1'}}
    torch.onnx.export(model, dummy_input, args.out,
                      input_names=input_names,
                      dynamic_axes=dynamic_axes,
                      output_names=output_names,
                      opset_version=11)


if __name__ == "__main__":
    ssl._create_default_https_context = ssl._create_unverified_context
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', help='onnx output name')
    parser.add_argument('--pth_path', help='model pth path')
    args = parser.parse_args()
    convert(args)