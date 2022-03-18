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

import torch
import torch.onnx
from inceptionv4_v2 import InceptionV4


def convert(pth_file_path, class_nums):
    checkpoint = torch.load(pth_file_path, map_location='cpu')
    model = InceptionV4(num_classes=class_nums)
    model.load_state_dict(checkpoint,False)
    model.eval()

    input_names = ["actual_input_1"]
    output_names = ["output1"]
    dummy_input = torch.randn(32, 3, 299, 299)
    torch.onnx.export(model, dummy_input, "InceptionV4_npu_32.onnx", input_names=input_names, output_names=output_names, opset_version=11)


if __name__ == "__main__":
    src_file_path = "model_best.pth.tar"
    class_num = 1001
    convert(src_file_path, class_num)
    