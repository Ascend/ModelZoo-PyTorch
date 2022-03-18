# Copyright (c) Soumith Chintala 2016,
# All rights reserved
#
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://spdx.org/licenses/BSD-3-Clause.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import argparse
import torch
import torch.onnx
from lib.config import config, update_config, hrnet
from collections import OrderedDict


def parse_args():
    parser = argparse.ArgumentParser(description='Train classification network')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=False,
                        type=str,
                        default='./experiments/cls_hrnet_w18_sgd_lr5e-2_wd1e-4_bs32_x100.yaml')
    
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--testModel',
                        help='testModel',
                        type=str,
                        default='')

    args = parser.parse_args()
    update_config(config, args)

    return args


def pth2onnx(input_file="", output_file=""):
    args = parse_args()

    checkpoint = torch.load("./output/imagenet/cls_hrnet_w18_sgd_lr5e-2_wd1e-4_bs32_x100/model_best.pth.tar", map_location='cpu')
    model = hrnet.get_cls_net(config)
    model.load_state_dict(checkpoint)
    model.eval()
    print(model)

    input_names = ["image_input"]
    output_names = ["output_1"]
    dynamic_axes = {'image': {0: '-1'}, 'class': {0: '-1'}}
    dummy_input = torch.randn(16, 3, 224, 224)
    torch.onnx.export(model, dummy_input, "hrnet_npu_16.onnx", input_names=input_names, dynamic_axes=dynamic_axes, output_names=output_names, verbose=True, opset_version=11)


if __name__ == "__main__":
    pth2onnx()
