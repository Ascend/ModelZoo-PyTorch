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


import sys
import argparse

import numpy as np
import torch
import mmcv
from mmcv.runner import load_checkpoint
from mmcv.onnx.symbolic import register_extra_symbolics
from mmaction.models import build_model

sys.path.append('mmaction2/tools')
from deployment.pytorch2onnx import _convert_batchnorm


def init_model(cfg_path, ckpt_path, is_localizer=False, softmax=True):

    cfg = mmcv.Config.fromfile(cfg_path, ckpt_path)

    if not is_localizer:
        cfg.model.backbone.pretrained = None

    # build the model
    model = build_model(
        cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    model = _convert_batchnorm(model)

    # onnx.export does not support kwargs
    if hasattr(model, 'forward_dummy'):
        import functools
        model.forward = functools.partial(model.forward_dummy, softmax=softmax)
    else:
        raise NotImplementedError(
            'Please implement the forward method for exporting.')
    load_checkpoint(model, ckpt_path, map_location='cpu')
    model.cpu().eval()

    return model


def pytorch2onnx(pytorch_model, output_file='tmp.onnx'):
    opset_version=11
    register_extra_symbolics(opset_version)

    input_shape = (1, 1, 3, 32, 224, 224)
    input_tensor = torch.randn(input_shape)
    input_names = ["video"]
    output_names = ["class"]
    dynamic_axes = {'video': {0: '-1'}, 'class': {0: '-1'}}
    torch.onnx.export(pytorch_model,
                     input_tensor,
                     output_file,
                     input_names=input_names,
                     output_names=output_names,
                     dynamic_axes=dynamic_axes,
                     export_params=True,
                     keep_initializers_as_inputs=True,
                     verbose=False,
                     opset_version=opset_version)
    print('Successfully exported ONNX model')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Convert MMAction2 models to ONNX')
    parser.add_argument('--config', type=str, help='test config file path')
    parser.add_argument('--checkpoint', type=str, help='checkpoint file')
    parser.add_argument('--output-file', type=str, default='tmp.onnx')
    parser.add_argument('--is-localizer', action='store_true',
                        help='whether it is a localizer')
    parser.add_argument('--softmax', action='store_true',
                        help='Add softmax layer at the end of recognizers')
    args = parser.parse_args()

    model = init_model(args.config, args.checkpoint, 
                       is_localizer=args.is_localizer, softmax=args.softmax)
    pytorch2onnx(model, output_file=args.output_file)
