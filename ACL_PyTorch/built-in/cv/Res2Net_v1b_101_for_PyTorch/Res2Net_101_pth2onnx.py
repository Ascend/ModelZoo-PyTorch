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


import itertools
import argparse
import os 
import sys
sys.path.append('./Res2Net-PretrainedModels')
from res2net_v1b import res2net101_v1b_26w_4s
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='res2net101_v1b inference')
    parser.add_argument('-m', '--trained_model', default=None,
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('-o', '--output', default=None,
                        type=str, help='ONNX model file')
    args = parser.parse_args()

    model = res2net101_v1b_26w_4s()
    checkpoint = torch.load(args.trained_model, map_location=torch.device('cpu'))

    model.load_state_dict(checkpoint)
    model.eval()

    inputs = torch.rand(1, 3, 224, 224)
    torch.onnx.export(model, inputs, args.output,
                    input_names=["x"], output_names=["output"],
                    dynamic_axes={"x": {0: "-1"}}, opset_version=11)
