# Copyright 2020 Huawei Technologies Co., Ltd
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

sys.path.append(r"./pytorch-nested-unet")
import torch
import torch.onnx
from archs import *


def convert():
    # https://github.com/4uiiurz1/pytorch-nested-unet
    model = NestedUNet(num_classes=1, input_channels=3, deep_supervision=False)
    checkpoint = torch.load(input_file, map_location="cpu")
    model.load_state_dict(checkpoint)

    model.eval()

    input_names = ["actual_input_1"]
    output_names = ["output1"]
    dynamic_axes = {'actual_input_1': {0: '-1'}, 'output1': {0: '-1'}}

    dummy_input = torch.randn(1, 3, 96, 96)

    torch.onnx.export(model, dummy_input, output_file, input_names=input_names, dynamic_axes=dynamic_axes,
                      output_names=output_names, opset_version=11, verbose=True)

    ts_model = torch.jit.trace(model, dummy_input)
    res = ts_model(dummy_input)
    print("res11111", res)
    # model.to_torchscript(method="trace", example_inputs=input_data)
    output_model = 'nested_unet.torchscript.pt'
    ts_model.save(output_model)
    print(f"FastPitch torch script model saved to {output_model}.")


if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    convert()