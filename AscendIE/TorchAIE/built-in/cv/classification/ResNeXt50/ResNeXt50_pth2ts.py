# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
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
    # https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    if (len(sys.argv) == 3):
        model = models.resnext50_32x4d(pretrained=False)
        checkpoint = torch.load(input_file, map_location=None)
        model.load_state_dict(checkpoint)
    else:
        model = models.resnext50_32x4d(pretrained=True)

    model.eval()
    input_data = torch.ones(1, 3, 224, 224)
    ts_model = torch.jit.trace(model, input_data)
    ts_model.save(output_file)
    print(f"Resnext50 torch script model saved to {output_file}.")

if __name__ == "__main__":
    if (len(sys.argv) == 3):
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    else:
        output_file = "./resnext50.ts"
    ssl._create_default_https_context = ssl._create_unverified_context
    convert()
