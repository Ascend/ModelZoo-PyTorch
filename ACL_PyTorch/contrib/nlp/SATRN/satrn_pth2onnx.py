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


from argparse import ArgumentParser
from functools import partial

import torch
from torch import nn

from mmocr.apis import init_detector
from mmcv.onnx import register_extra_symbolics


def _convert_batchnorm(module):
    module_output = module
    if isinstance(module, torch.nn.SyncBatchNorm):
        module_output = torch.nn.BatchNorm2d(
            module.num_features, module.eps, module.momentum, 
            module.affine, module.track_running_stats
        )
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
            # keep requires_grad unchanged
            module_output.weight.requires_grad = module.weight.requires_grad
            module_output.bias.requires_grad = module.bias.requires_grad
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        module_output.add_module(name, _convert_batchnorm(child))
    del module
    return module_output


def pytorch2onnx(model, output_file, opset_version=11):

    device = torch.device(type='cpu')
    model.to(device).eval()
    _convert_batchnorm(model)

    img_metas = [[{
        'img_shape': (32, 100, 3), 
        'valid_ratio': 1.0, 
        'resize_shape': (32, 100, 3), 
        'img_norm_cfg': {
            'mean': [0.485, 0.456, 0.406], 
            'std': [0.229, 0.224, 0.225]
        }, 
    }]]
    
    model.forward = partial(
        model.forward,
        img_metas=img_metas,
        return_loss=False,
        rescale=True)

    register_extra_symbolics(opset_version)
    dynamic_axes = {'input': {0: '-1'}, 'output': {0: '-1'}}
    with torch.no_grad():
        torch.onnx.export(
            model,
            torch.randn(1, 3, 32, 100),
            output_file,
            input_names=['input'],
            output_names=['output'],
            export_params=True,
            keep_initializers_as_inputs=False,
            verbose=False,
            opset_version=opset_version,
            dynamic_axes=dynamic_axes)
    print(f'Successfully exported ONNX model: {output_file}')


def main():
    parser = ArgumentParser(
                description='Convert models from pytorch to ONNX')
    parser.add_argument('--cfg-path', type=str, required=True, 
                        help='path ot config file.')
    parser.add_argument('--ckpt-path', type=str,  required=True, 
                        help='path to checkpoint.')
    parser.add_argument('--onnx-path', type=str,  required=True, 
                        help='path to save onnx model.')
    parser.add_argument('--opset-version', type=int, default=11, 
                        help='ONNX opset version, default to 11.')
    args = parser.parse_args()


    device = torch.device(type='cpu')
    model = init_detector(args.cfg_path, args.ckpt_path, device=device)
    if hasattr(model, 'module'):
        model = model.module

    pytorch2onnx(model, args.onnx_path, args.opset_version)


if __name__ == '__main__':
    main()
