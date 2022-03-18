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
import onnx
import torch.onnx
sys.path.append(r"./pytorch-ssd")
from vision.ssd.vgg_ssd import create_vgg_ssd

def pth2onnx(input_file, output_file):
    # https://github.com/qfgaohao/pytorch-ssd
    model = create_vgg_ssd(21, is_test=True)
    model.load(input_file)

    model.eval()
    input_names = ["actual_input_1"]
    output_names = ["scores", "boxes"]
    dynamic_axes = {'actual_input_1':{0:'-1'}, 'scores':{0:'-1'}, 'boxes':{0:'-1'}}
    dummy_input = torch.randn(1, 3, 300, 300)

    torch.onnx.export(model, dummy_input, output_file, verbose=False, input_names=input_names, output_names=output_names, dynamic_axes = dynamic_axes, opset_version=11)

if __name__ == "__main__":
    if (len(sys.argv) == 3):
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        pth2onnx(input_file, output_file)
