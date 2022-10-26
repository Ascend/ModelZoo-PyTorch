# -*- coding: utf-8 -*-
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

import mmcv
import torch
from mmcv.runner import load_checkpoint
from mmpose.models import build_posenet
from mmcv.onnx.symbolic import register_extra_symbolics   


def _convert_batchnorm(module):
    """Convert the syncBNs into normal BN3ds."""
    module_output = module
    if isinstance(module, torch.nn.SyncBatchNorm):
        module_output = torch.nn.BatchNorm3d(module.num_features, module.eps,
                                             module.momentum, module.affine,
                                             module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
            module_output.weight.requires_grad = module.weight.requires_grad
            module_output.bias.requires_grad = module.bias.requires_grad
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        module_output.add_module(name, _convert_batchnorm(child))
    del module
    return module_output


def build_hourglass(cfg_file, pth_file):
    cfg = mmcv.Config.fromfile(cfg_file)

    model = build_posenet(cfg.model)
    model = _convert_batchnorm(model)

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'Please implement the forward method for exporting.')

    load_checkpoint(model, pth_file, map_location='cpu')
    return model


def pth2onnx(cfg_file,
            pth_file,
            input_shape=[32, 3, 384, 384],
            output_file='hourglass.onnx'):

    model = build_hourglass(cfg_file, pth_file)
    model.cpu().eval()

    one_img = torch.randn(input_shape).float()

    register_extra_symbolics(11) # MMPose only supports opset 11
    print("===============start to export onnx===============")
    torch.onnx.export(
        model,
        one_img,
        output_file,
        export_params=True,
        keep_initializers_as_inputs=True,
        verbose=False,
        opset_version=11)
    print(f'Successfully exported ONNX model: {output_file}')

