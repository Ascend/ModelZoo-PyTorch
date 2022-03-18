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

import sys

sys.path.append(r"./TokenLabeling")

import torch
# import argparse
from timm.models import create_model, apply_test_time_pool, load_checkpoint, is_model, list_models
import tlt.models

import os
import numpy as np

from timm.data.transforms_factory import transforms_imagenet_eval
from torchvision import transforms
from PIL import Image


# parser = argparse.ArgumentParser()
# parser.add_argument('--model', type=str, default='lvvit_s')
# parser.add_argument('--use-ema', dest='use_ema', action='store_true',
#                     help='use ema version of weights if present')
# parser.add_argument('--checkpoint', type=str, default='')
# parser.add_argument('--pretrained', dest='pretrained', action='store_true',
#                     help='use pre-trained model')
# parser.add_argument('--gp', default=None, type=str, metavar='POOL',
#                     help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
# parser.add_argument('--output_file', default='lvvit_s.onnx', type=str)
# parser.add_argument('-b', '--batch_size', default=16, type=int)


def main():
    if not os.path.exists('./model'):
        os.mkdir('./model')

    device = torch.device('cpu')
    input_names = ["image"]
    output_names = ["features"]
    dynamic_axes = {'image': {0: f'{sys.argv[3]}'}, 'features': {0: f'{sys.argv[3]}'}}
    model = create_model(
        'lvvit_s',
        pretrained=False,
        num_classes=None,
        in_chans=3,
        global_pool=None,
        scriptable=False,
        img_size=224)
    # model.cuda()
    # load_checkpoint(model, args.checkpoint, args.use_ema, strict=False)
    load_checkpoint(model, sys.argv[1], False, strict=False)
    model.to(device)
    model.eval()
    dummy_input = torch.randn(int(sys.argv[3]), 3, 224, 224, device='cpu')
    torch.onnx.export(model,
                      dummy_input,
                      sys.argv[2],
                      input_names=input_names,
                      output_names=output_names,
                      dynamic_axes=dynamic_axes,
                      opset_version=13,
                      verbose=True)


main()

