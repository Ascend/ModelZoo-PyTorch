# Copyright 2022 Huawei Technologies Co., Ltd
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
# ============================================================================

import torch
import torch.onnx
from collections import OrderedDict
import argparse
from mmcv import Config, DictAction
from mmpose.models import build_posenet
from mmcv.runner import load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description='Train a pose model')
    parser.add_argument('--config',default="configs/top_down/deeppose/coco/npu_deeppose_res50_coco_256x192.py",help='train config file path')
    parser.add_argument('--checkpoint',default="./epoch_210.pth",help='train model pth')
    args = parser.parse_args()
    return args
    

def convert():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    model = build_posenet(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    # onnx.export does not support kwargs
    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'Please implement the forward method for exporting.')

    model.eval()
    input_names = ["actual_input_1"]
    output_names = ["output1"]
    x = torch.randn(1, 3, 256, 192)
    torch.onnx.export(model, x, "deeppose_npu.onnx", input_names=input_names, output_names=output_names,
                      opset_version=11)


if __name__ == "__main__":
    convert()
