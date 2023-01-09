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

import torch
import argparse
import numpy as np
import sys
sys.path.append("./YOLOF")
from cvpods.engine import default_setup
from cvpods.checkpoint import DefaultCheckpointer
sys.path.append("./YOLOF/playground/detection/coco/yolof/yolof.cspdarknet53.DC5.9x/")
sys.path.append("./YOLOF/playground/detection/coco/yolof/")
from net import build_model
from config import config

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def convert_batchnorm(module, process_group=None):
    module_output = module
    if isinstance(module, torch.nn.SyncBatchNorm):
        module_output = torch.nn.BatchNorm2d(num_features=module.num_features,
                                             eps=module.eps,
                                             momentum=module.momentum,
                                             affine=module.affine,
                                             track_running_stats=module.track_running_stats)
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        module_output.add_module(name, convert_batchnorm(child, process_group))
    del module
    return module_output


def pth2onnx(args, fake_input, opts):
    config.merge_from_list(opts)
    model = build_model(config)
    model._batch_size = args.batch_size
    model.forward = model.forward_onnx
    model = convert_batchnorm(model)
    model.eval()
    DefaultCheckpointer(model, save_dir=config.OUTPUT_DIR).resume_or_load(
        config.MODEL.WEIGHTS, resume=False
    )
    torch.onnx.export(model, fake_input, args.out, verbose=True, opset_version=11,
                      input_names=['input'], enable_onnx_checker=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', help='model config path',
                        default="YOLOF/playground/detection/coco/yolof/yolof.cspdarknet53.DC5.9x")
    parser.add_argument('--out', help='onnx output name', default="yolof.onnx")
    parser.add_argument('--pth_path', help='model pth path', default="./YOLOF_CSP_D_53_DC5_9x.pth")
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()

    img_shape = (args.batch_size, 3, 608, 608)
    fake_input = torch.randn(*img_shape)
    opts = ['MODEL.WEIGHTS', args.pth_path, "MODEL.DEVICE", "cpu"]
    pth2onnx(args, fake_input, opts)
