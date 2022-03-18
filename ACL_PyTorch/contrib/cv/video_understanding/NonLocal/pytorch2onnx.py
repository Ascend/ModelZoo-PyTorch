"""
Copyright 2020 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
============================================================================
"""
# !/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import mmcv
import numpy as np
import torch
from mmcv.runner import load_checkpoint

from mmaction.models import build_model

try:
    import onnx
    import onnxruntime as rt
except ImportError as e:
    raise ImportError('Please install onnx and onnxruntime first.')

try:
    from mmcv.onnx.symbolic import register_extra_symbolics
except ImportError as e:
    raise ImportError('Please install mmcv.')


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


def pytorch2onnx(model_,
                 input_shape,
                 opset_version=11,
                 show=False,
                 output_file='tmp.onnx',
                 verify=False):
    """Convert pytorch model to onnx model.

    Args:
        model_ (:obj:`nn.Module`): The pytorch model to be exported.
        input_shape (tuple[int]): The input tensor shape of the model.
        opset_version (int): Opset version of onnx used. Default: 11.
        show (bool): Determines whether to print the onnx model architecture.
            Default: False.
        output_file (str): Output onnx model name. Default: 'tmp.onnx'.
        verify (bool): Determines whether to verify the onnx model.
            Default: False.
    """
    model_.cpu().eval()

    input_tensor = torch.randn(input_shape)

    register_extra_symbolics(opset_version)
    input_names = ["video"]
    output_names = ["class"]
    dynamic_axes = {'video': {0: '-1'}, 'class': {0: '-1'}}
    torch.onnx.export(
        model_,
        input_tensor,
        output_file,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        export_params=True,
        keep_initializers_as_inputs=True,
        verbose=show,
        opset_version=opset_version)

    print('Successfully exported ONNX model')
    if verify:
        # check by onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)

        # check the numerical value
        # get pytorch output
        pytorch_result = model_(input_tensor)[0].detach().numpy()

        # get onnx output
        input_all = [node.name for node in onnx_model.graph.input]
        input_initializer = [
            node.name for node in onnx_model.graph.initializer
        ]
        net_feed_input = list(set(input_all) - set(input_initializer))
        assert len(net_feed_input) == 1
        sess = rt.InferenceSession(output_file)
        onnx_result = sess.run(
            None, {net_feed_input[0]: input_tensor.detach().numpy()})[0]
        # only compare part of results
        random_class = np.random.randint(pytorch_result.shape[1])
        assert np.allclose(
            pytorch_result[:, random_class], onnx_result[:, random_class]
        ), 'The outputs are different between Pytorch and ONNX'
        print('The numerical values are same between Pytorch and ONNX')


def parse_args():
    """
    input argument receiving function
    :return: input arguments
    """
    parser = argparse.ArgumentParser(
        description='Convert MMAction2 models to ONNX')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--show', action='store_true', help='show onnx graph')
    parser.add_argument('--output-file', type=str, default='tmp.onnx')
    parser.add_argument('--opset-version', type=int, default=11)
    parser.add_argument(
        '--verify',
        action='store_true',
        help='verify the onnx model output against pytorch output')
    parser.add_argument(
        '--is-localizer',
        action='store_true',
        help='whether it is a localizer')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[1, 3, 8, 224, 224],
        help='input video size')
    parser.add_argument(
        '--softmax',
        action='store_true',
        help='wheter to add softmax layer at the end of recognizers')
    args_ = parser.parse_args()
    return args_


if __name__ == '__main__':
    args = parse_args()

    assert args.opset_version == 11, 'MMAction2 only supports opset 11 now'

    cfg = mmcv.Config.fromfile(args.config)
    # import modules from string list.

    if not args.is_localizer:
        cfg.model.backbone.pretrained = None

    # build the model
    model = build_model(
        cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    model = _convert_batchnorm(model)

    # onnx.export does not support kwargs
    if hasattr(model, 'forward_dummy'):
        import functools
        model.forward = functools.partial(model.forward_dummy, softmax=args.softmax)
    else:
        raise NotImplementedError(
            'Please implement the forward method for exporting.')

    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    # conver model to onnx file
    pytorch2onnx(
        model,
        args.shape,
        opset_version=args.opset_version,
        show=args.show,
        output_file=args.output_file,
        verify=args.verify)
