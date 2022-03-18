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
sys.path.append('slowfast')
from slowfast.models import build_model
from slowfast.utils import checkpoint as cu
from slowfast.utils.parser import load_config, parse_args
from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.misc import launch_job
def perform_x3d_pth2onnx(output_path, cfg):
    model = build_model(cfg)
    cu.load_test_checkpoint(cfg, model)
    model.eval()
    input_names = ["image"]
    output_names = ["class"]
    dynamic_axes = {'image': {0: '-1'}, 'class': {0: '-1'}}
    dummy_input = [torch.randn(16, 3, 13, 182, 182)]
    torch.onnx.export(model, dummy_input, output_path, input_names = input_names, dynamic_axes = dynamic_axes, output_names = output_names, opset_version=11, verbose=True)

def x3d_pth2onnx(cfg):
    output_path = cfg.X3D_PTH2ONNX.ONNX_OUTPUT_PATH
    perform_x3d_pth2onnx(output_path, cfg)

if __name__== '__main__':
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)

    launch_job(cfg=cfg, init_method=args.init_method, func=x3d_pth2onnx)
