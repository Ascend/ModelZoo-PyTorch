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

import sys
import ssl
import torch
import torch.onnx
import torchvision.models as models

def convert():
    # https://github.com/pytorch/vision/blob/master/torchvision/models/googlenet.py
    if (len(sys.argv) == 3):
        model = models.googlenet(pretrained=False, transform_input=True, aux_logits=True, init_weights=False)
        checkpoint = torch.load(input_file, map_location=None)
        model.load_state_dict(checkpoint)
        model.aux_logits = False
        model.aux1 = None
        model.aux2 = None
    else:
        model = models.googlenet(pretrained=True)

    model.eval()
    input_names = ["actual_input_1"]
    output_names = ["output1"]
    dynamic_axes = {'actual_input_1': {0: '-1'}, 'output1': {0: '-1'}}

    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(model, dummy_input, output_file, input_names = input_names, dynamic_axes = dynamic_axes, output_names = output_names, opset_version=11)

if __name__ == "__main__":
    if (len(sys.argv) == 3):
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    else:
        output_file = "./googlenet.onnx"
    ssl._create_default_https_context = ssl._create_unverified_context
    convert()
