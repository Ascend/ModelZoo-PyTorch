# Copyright 2023 Huawei Technologies Co., Ltd
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
# ============================================================================

# -*- coding: utf-8 -*-
import sys
import argparse
import torch
import torchvision
from SCNet.scnet import scnet50_v1d


def pth2onnx(input_file, output_file):
    # load net
    model = scnet50_v1d(pretrained=False)  # initialize
    checkpoint = torch.load(input_file, map_location=None)
    model.load_state_dict(checkpoint)
    model.eval()

    input_names = ['input']
    output_names = ['output']
    x = torch.randn(1, 3, 224, 224)
    dynamic_axes = {'input': {0: '-1'}}
    torch.onnx.export(model, x, output_file, input_names=input_names, output_names=output_names,
                      opset_version=11, verbose=True, dynamic_axes=dynamic_axes)


if __name__ == '__main__':
    input_f = sys.argv[1]
    output_f = sys.argv[2]
    pth2onnx(input_f, output_f)
