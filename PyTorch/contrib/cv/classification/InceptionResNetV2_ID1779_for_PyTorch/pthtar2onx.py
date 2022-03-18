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
# limitations under the License.


import torch
import torch.onnx

from inceptionresnetv2 import InceptionResNetV2


def convert():
    checkpoint = torch.load("checkpoint.pth.tar", map_location='cpu')
    model = InceptionResNetV2()
    model.load_state_dict(checkpoint, False)
    model.eval()

    input_names = ["actual_input_1"]
    output_names = ["output1"]
    dummy_input = torch.randn(32, 3, 299, 299)
    torch.onnx.export(model, dummy_input, "InceptionResNetV2_npu_16.onnx", input_names=input_names, output_names=output_names, opset_version=11)


if __name__ == "__main__":
    convert()


