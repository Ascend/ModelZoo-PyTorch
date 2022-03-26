# Copyright 2022 Huawei Technologies Co., Ltd
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

import sys
import torch
import mnasnet
import torch.onnx

from collections import OrderedDict


def convert():
    checkpoint = torch.load(input_file, map_location=None)
    model = mnasnet.mnasnet1_0()
    model.load_state_dict(checkpoint)
    model.eval()
    print(model)

    input_names = ["image"]
    output_names = ["class"]
    dynamic_axes = {'image': {0: '-1'}, 'class': {0: '-1'}}
    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(model, dummy_input, "mnasnet1_0.onnx", input_names=input_names, output_names=output_names,
                      opset_version=11)


if __name__ == "__main__":
    input_file = sys.argv[1]
    convert()
