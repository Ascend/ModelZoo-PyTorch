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
sys.path.append(r"./TokenLabeling")

import torch
from timm.models import create_model, apply_test_time_pool, load_checkpoint, is_model, list_models
import tlt.models

import os
import numpy as np

from timm.data.transforms_factory import transforms_imagenet_eval
from torchvision import transforms
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default="")
parser.add_argument('--onnx_path', type=str, default="")
parser.add_argument('--batch_size', type=int, default=1)
opt = parser.parse_args()


def main():
    if not os.path.exists('./model'):
        os.mkdir('./model')

    device = torch.device('cpu')
    input_names = ["image"]
    output_names = ["features"]
    dynamic_axes = {'image': {0: '-1'}, 'features': {0: '-1'}}
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
    load_checkpoint(model, opt.model_path, False, strict=False)
    model.to(device)
    model.eval()
    dummy_input = torch.randn(int(opt.batch_size), 3, 224, 224, device='cpu')
    torch.onnx.export(model,
                      dummy_input,
                      opt.onnx_path,
                      input_names=input_names,
                      output_names=output_names,
                      dynamic_axes=dynamic_axes,
                      opset_version=13,
                      verbose=True)


main()

