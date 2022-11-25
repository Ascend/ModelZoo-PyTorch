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
import torchvision.models as models
import torch.nn as nn
import argparse


def convert(args):
    model = models.resnet50(pretrained=False)
    model.fc = nn.Identity()
    state_dict = torch.load(args.backbone_pth, map_location='cpu')
    model.load_state_dict(state_dict, strict=True)

    linear_weights = torch.load(args.linear_pth,
                                map_location='cpu')["state_dict"]
    linear = nn.Linear(2048, 1000)
    linear.weight.data = linear_weights['module.linear.weight']
    linear.bias.data = linear_weights['module.linear.bias']
    model.fc = linear
    model.eval()

    input_names = ["input"]
    output_names = ["output"]
    dummy_input = torch.randn(16, 3, 224, 224)
    dynamic_axes = {'input': {0: '-1'}, 'output': {0: '-1'}}
    torch.onnx.export(model, dummy_input, args.out,
                      input_names=input_names,
                      dynamic_axes=dynamic_axes,
                      output_names=output_names,
                      opset_version=11)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', help='onnx output name')
    parser.add_argument('--backbone_pth', help='backbone model pth path')
    parser.add_argument('--linear_pth', help='linear model pth path')
    args = parser.parse_args()
    convert(args)
