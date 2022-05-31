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
sys.path.append(r"./Cross-Scale-Non-Local-Attention/src/")
from model.csnln import CSNLN
from option import args
from collections import OrderedDict


def pth2onnx(input_file, output_file):
    model = CSNLN(args)
    model.load_state_dict(torch.load(
        input_file, map_location=torch.device('cpu')), strict=False)

    model.eval()
    dummy_input = torch.randn(1, 3, 56, 56)

    torch.onnx.export(model, dummy_input, output_file, opset_version=11, verbose=False)


if __name__ == "__main__":
    input_file = args.pre_train
    output_file = args.save
    pth2onnx(input_file, output_file)
