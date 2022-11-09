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
# ============================================================================
"""Convert pth to onnx module"""

import argparse
from functools import partial

import numpy as np
import onnx

import torch
import torch._C
import torch.serialization
from torch import nn

import mmcv
from mmcv.onnx import register_extra_symbolics
from mmcv.runner import load_checkpoint

from mmseg.models import build_segmentor

torch.manual_seed(3)


def _convert_batchnorm(module):
    module_output = module
    if isinstance(module, torch.nn.SyncBatchNorm):
        module_output = torch.nn.BatchNorm2d(module.num_features, module.eps,
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


def _demo_mm_inputs(input_shape, num_classes):
    """Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple):
            input batch dimensions
        num_classes (int):
            number of semantic classes
    """
    (N, C, H, W) = input_shape
    rng = np.random.RandomState(0)
    imgs = rng.rand(*input_shape)
    segs = rng.randint(
        low=0, high=num_classes - 1, size=(N, 1, H, W)).astype(np.uint8)
    img_metas = [{
        'img_shape': (H, W, C),
        'ori_shape': (H, W, C),
        'pad_shape': (H, W, C),
        'filename': '<demo>.png',
        'scale_factor': 1.0,
        'flip': False,
    } for _ in range(N)]
    mm_inputs = {
        'imgs': torch.FloatTensor(imgs).requires_grad_(True),
        'img_metas': img_metas,
        'gt_semantic_seg': torch.LongTensor(segs)
    }
    return mm_inputs


def pytorch2onnx(model,
                 input_shape,
                 batch_size=1,
                 opset_version=11,
                 output_file='seMask.onnx'):
    """Export Pytorch model to ONNX model and verify the outputs are same
    between Pytorch and ONNX.

    Args:
        model (nn.Module): Pytorch model we want to export.
        input_shape (tuple): Use this input shape to construct
            the corresponding dummy input and execute the model.
        batch_size (int): Model batch size.
        opset_version (int): The onnx op version. Default: 11.
        output_file (string): The path to where we store the output ONNX model.
            Default: `seMask.onnx`.
    """
    model.cpu().eval()

    if isinstance(model.decode_head, nn.ModuleList):
        num_classes = model.decode_head[-1].num_classes
    else:
        num_classes = model.decode_head.num_classes

    mm_inputs = _demo_mm_inputs(input_shape, num_classes)

    img_metas = mm_inputs.pop('img_metas')

    img_meta_list = [[img_meta] for img_meta in img_metas]

    # replace original forward function
    origin_forward = model.forward
    model.forward = partial(
        model.forward, img_metas=img_meta_list, return_loss=False)

    input_names = ['image']
    output_names = ['output']
    dummy_input = torch.randn(batch_size, 3, 1024, 2048)

    torch.onnx.export(
        model,
        [dummy_input],
        output_file,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        opset_version=opset_version,
        do_constant_folding=True)
    print(f'Successfully exported ONNX model: {output_file}')
    # model.forward = origin_forward

    # onnx_model = onnx.load(output_file)
    # model_simp, check = simplify(onnx_model)
    # assert check, "Simplified ONNX model could not be calidated"
    # onnx.save(model_simp, output_file)
    # print(f'Successfully exported simplified ONNX model: {output_file}')


def parse_args():
    """Pth to onnx arguments"""
    parser = argparse.ArgumentParser(description='Convert MMSeg to ONNX')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file',
                        default='semask_small_fpn_cityscapes.pth')
    parser.add_argument('--batch_size', type=int, help='model batch size', default=1)
    parser.add_argument(
        '--verify', action='store_true', help='verify the onnx model')
    parser.add_argument('--output_file', type=str, default='SeMask.onnx')
    parser.add_argument('--opset-version', type=int, default=11)
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[1024, 2048],
        help='input image size')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (
            1,
            3,
        ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    segmentor = build_segmentor(
        cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    # convert SyncBN to BN
    segmentor = _convert_batchnorm(segmentor)

    if args.checkpoint:
        load_checkpoint(segmentor, args.checkpoint, map_location='cpu')

    # conver model to onnx file
    pytorch2onnx(
        segmentor,
        input_shape,
        batch_size = args.batch_size,
        opset_version=args.opset_version,
        output_file=args.output_file)
