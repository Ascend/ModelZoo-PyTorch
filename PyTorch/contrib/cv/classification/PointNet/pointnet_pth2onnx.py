# BSD 3-Clause License

# Copyright (c) Soumith Chintala 2016,
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import torch
import torch.nn.parallel
import torch.utils.data
from pointnet.model import PointNetCls


def pth2onnx(opt):
    classifier = PointNetCls(k=opt.num_classes, feature_transform=opt.feature_transform, device=opt.device)
    if opt.model != '':
        classifier.load_state_dict(torch.load(opt.model, map_location='cpu')['model_state_dict'])
    classifier.eval()

    input_names = ["image"]
    output_names = ["class"]
    dynamic_axes = {'image': {0: '-1'}, 'class': {0: '-1'}}

    dummy_input = torch.randn(32, 3, 2500)

    torch.onnx.export(
        classifier, dummy_input, opt.output_file, dynamic_axes=dynamic_axes,
        input_names=input_names, output_names=output_names, verbose=True, opset_version=11)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--model', type=str, default='./checkpoint_79_epoch.pkl', help='model path')
    parser.add_argument('--output_file', type=str, default='./pointnet.onnx', help='output path')
    parser.add_argument('--feature_transform', type=bool, default=True, help="use feature transform")
    opt = parser.parse_args()
    pth2onnx(opt)
