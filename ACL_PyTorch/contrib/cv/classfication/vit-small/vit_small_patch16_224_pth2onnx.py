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
import sys
sys.path.append(r'./pytorch-image-models')
import timm

def pth2onnx(model, output_file):
    model.eval()
    input_names = ["image"]
    output_names = ["class"]
    dynamic_axes = {'image': {0: '-1'}, 'class': {0: '-1'}}
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy_input, output_file, input_names = input_names, dynamic_axes = dynamic_axes, output_names = output_names, opset_version=11, verbose=False)
    print("save to {}".format(output_file))

if __name__ == "__main__":
    model = timm.create_model('vit_small_patch16_224', pretrained=False, checkpoint_path=sys.argv[1])
    pth2onnx(model, sys.argv[2])
