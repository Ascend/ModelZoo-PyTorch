# -*- coding: utf-8 -*-
# Copyright (c) Soumith Chintala 2016,
# All rights reserved
#
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://spdx.org/licenses/BSD-3-Clause.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import torch
import sys
from van import *

if __name__ == '__main__':
    model = van_b2(pretrained=False)

    onnxpath = sys.argv[1]
    checkpoint = torch.load(sys.argv[2], map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["state_dict"])

    model.eval()
    input_names = ["image"]
    output_names = ["class"]
    dynamic_axes = {'image': {0: '-1'}, 'class': {0: '-1'}}
    input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, input, onnxpath, input_names=input_names,
                      dynamic_axes=dynamic_axes, output_names = output_names, opset_version=11, verbose=True)

