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


import argparse
from functools import partial

import torch
from torch import nn

import mmcv
from mmcv.onnx import register_extra_symbolics
from mmcv.runner import load_checkpoint
from mmseg.models import build_segmentor

torch.manual_seed(3)


def parse_args():
    parser = argparse.ArgumentParser(
                        description='Convert MMSeg to ONNX')
    parser.add_argument('--config', type=str, required=True, 
                        help='model config file path')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='checkpoint file')
    parser.add_argument('--onnx', type=str, required=True,
                        help='path to save onnx model')
    parser.add_argument('--batchsize', type=int, default=1,
                        help='how many samples are processed at a time')
    parser.add_argument('--opset-version', type=int, default=11,
                        help='onnx opset version.')
    args = parser.parse_args()
    return args


def _convert_batchnorm(module):
    module_output = module
    if isinstance(module, torch.nn.SyncBatchNorm):
        module_output = torch.nn.BatchNorm2d(
            module.num_features, module.eps,
            module.momentum, module.affine,
            module.track_running_stats
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


def pytorch2onnx(model, dummy_input, output_file, opset_version=11):

    model.cpu().eval()
    model.forward = partial(model.encode_decode, img_metas=None)
    register_extra_symbolics(opset_version)
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            output_file,
            input_names=['input'],
            output_names=['output'],
            export_params=True,
            keep_initializers_as_inputs=False,
            dynamic_axes={},
            verbose=False,
            opset_version=opset_version,
        )
    print(f'Successfully exported ONNX model: {output_file}')
 




def main():

    args = parse_args()
    assert args.batchsize > 0

    cfg = mmcv.Config.fromfile(args.config)
    cfg.merge_from_dict({
        'model.test_cfg.mode': 'slide', 
        'model.test_cfg.crop_size': (512, 512), 
        'model.test_cfg.stride': (384, 384)
    })
    cfg.model.pretrained = None
    cfg.model.train_cfg = None

    # build the model and load checkpoint
    segmentor = build_segmentor(cfg.model, train_cfg=None, 
                                test_cfg=cfg.get('test_cfg'))
    # convert SyncBN to BN
    segmentor = _convert_batchnorm(segmentor)
    # load checkpoint
    checkpoint = load_checkpoint(segmentor, args.checkpoint, map_location='cpu')

    # convert model to onnx file
    pytorch2onnx(
        segmentor,
        torch.randn(args.batchsize, 3, 512, 512),
        args.onnx,
        opset_version=args.opset_version
    )


if __name__ == '__main__':
    main()
