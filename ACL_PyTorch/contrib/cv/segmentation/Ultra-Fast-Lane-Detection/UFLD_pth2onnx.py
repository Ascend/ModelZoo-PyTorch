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


import argparse
from enum import Enum

import onnx
import torch

from model.model import parsingNet


class ModelType(Enum):
    TUSIMPLE = 0
    CULANE = 1


class ModelConfig():

    def __init__(self, model_type):

        if model_type == ModelType.TUSIMPLE:
            self.init_tusimple_config()
        else:
            self.init_culane_config()

    def init_tusimple_config(self):
        self.img_w = 1280
        self.img_h = 720
        self.griding_num = 100
        self.cls_num_per_lane = 56

    def init_culane_config(self):
        self.img_w = 1640
        self.img_h = 590
        self.griding_num = 200
        self.cls_num_per_lane = 18


def convert_model(model_path, onnx_file_path, model_type=ModelType.TUSIMPLE):
    # Load model configuration based on the model type
    cfg = ModelConfig(model_type)

    # Load the model architecture
    net = parsingNet(pretrained=False, backbone='18', 
                     cls_dim=(cfg.griding_num + 1, cfg.cls_num_per_lane, 4),
                     use_aux=False)

    state_dict = torch.load(model_path, map_location='cpu')['model']

    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    # Load the weights into the model
    net.load_state_dict(compatible_state_dict, strict=False)

    img = torch.zeros(1, 3, 288, 800).to('cpu')
    input_name =['input']
    output_name =['output']

    dynamic_axes = {'input':{0:'-1'}, 'output':{0:'-1'}}
    torch.onnx.export(net, img, onnx_file_path,
                      input_names=input_name,
                      dynamic_axes=dynamic_axes, 
                      output_names=output_name, 
                      verbose=False)

    # Check that the IR is well formed
    model = onnx.load(onnx_file_path)

    onnx.checker.check_model(model)
    # Print a human readable representation of the graph


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        'convert original image to bin file.')
    parser.add_argument('--model-path', type=str, required=True,
                        help='path to weights file.')
    parser.add_argument('--onnx-path', type=str, required=True,
                        help='path to save onnx file.')
    parser.add_argument('--model-type', type=int, default=0, choices=[0, 1],
                        help='choice a dataset. {0: Tisimple, 1: Culane}.')
    args = parser.parse_args()

    convert_model(args.model_path, args.onnx_path, ModelType(args.model_type))
    print('ONNX generated.')
