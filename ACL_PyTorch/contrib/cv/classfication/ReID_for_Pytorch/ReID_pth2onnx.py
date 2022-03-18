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

import os
import sys
import argparse
import torch
import torch.onnx
sys.path.append('./reid-strong-baseline')
from config import cfg
from modeling import build_model

from collections import OrderedDict
def proc_nodes_module(checkpoint):
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        if "classifier" in k:
            continue
        new_state_dict[k] = v
    return new_state_dict

def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Inference")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    num_classes = 751
    model = build_model(cfg, num_classes)
    checkpoint = torch.load(cfg.TEST.WEIGHT, map_location='cpu')
    #checkpoint = proc_nodes_module(checkpoint)
    model.load_state_dict(checkpoint)
    model.eval()

    input_names = ["image"]
    output_names = ["class"]
    dynamic_axes = {'image': {0: '-1'}, 'class': {0: '-1'}}
    dummy_input = torch.randn(1, 3, 256, 128)
    export_onnx_file = "ReID.onnx"

    torch.onnx.export(model, dummy_input, export_onnx_file, input_names=input_names, dynamic_axes=dynamic_axes,
                      output_names=output_names, opset_version=11, verbose=True)

if __name__ == '__main__':
    main()
