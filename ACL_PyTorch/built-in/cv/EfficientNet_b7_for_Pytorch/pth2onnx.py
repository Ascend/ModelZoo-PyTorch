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
import argparse

import torch
from efficientnet_pytorch import EfficientNet

def export(version):
    params_dict = {
            # Coefficients:     width,depth,res,dropout
            'efficientnet-b0': (1.0, 1.0, 224, 0.2),
            'efficientnet-b1': (1.0, 1.1, 240, 0.2),
            'efficientnet-b2': (1.1, 1.2, 260, 0.3),
            'efficientnet-b3': (1.2, 1.4, 300, 0.3),
            'efficientnet-b4': (1.4, 1.8, 380, 0.4),
            'efficientnet-b5': (1.6, 2.2, 456, 0.4),
            'efficientnet-b6': (1.8, 2.6, 528, 0.5),
            'efficientnet-b7': (2.0, 3.1, 600, 0.5),
            'efficientnet-b8': (2.2, 3.6, 672, 0.5),
        }

    model_name = 'efficientnet-b%s' % version
    model = EfficientNet.from_pretrained(model_name)
    _, _, img_size, _ = params_dict.get(model_name)
    dummy_input = torch.randn(1, 3, img_size, img_size)

    model.set_swish(memory_efficient=False)
    torch.onnx.export(model,
                    dummy_input,
                    "efficientnet_b%s_%s_%d.onnx" % (version, 'dym', img_size),
                    input_names=['image'],
                    output_names=['output'],
                    dynamic_axes={
                        "image": {0: "bs"},
                    },
                    verbose=True,
                    opset_version=11)

if __name__ == "__main__":
    """
    python3.7 pth2onnx.py --version=7
    """

    parser = argparse.ArgumentParser(description='EfficientNet export onnx')
    parser.add_argument('--version', type=str, default=7, help='efficientnet version: 0~7')
    args = parser.parse_args()

    export(args.version)
