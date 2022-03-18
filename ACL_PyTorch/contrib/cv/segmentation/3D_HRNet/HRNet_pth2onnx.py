# Copyright 2020 Huawei Technologies Co., Ltd
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
import torch.onnx
import argparse

import sys
sys.path.append('HRNet-Semantic-Segmentation/lib')

from models import *
from config import config
from config import update_config

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    parser.add_argument('--cfg',
                        default='HRNet-Semantic-Segmentation/experiments/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml',
                        help='experiment configure file name',
                        type=str)
    parser.add_argument('opts',
                    help="Modify config options using the command-line",
                    default=None,
                    nargs=argparse.REMAINDER)
    parser.add_argument('--pth', default='best.pth', help='load pth file')
    args = parser.parse_args()
    update_config(config, args)
    return args

def proc_node_module(checkpoint):
    new_state_dict = {}
    for k, v in checkpoint.items():
        if "model." in k:
            name = k.replace("model.", "")
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict


def convert():
    args = parse_args()
    checkpoint = torch.load(args.pth, map_location='cpu')
    checkpoint = proc_node_module(checkpoint)
    model = models.seg_hrnet_ocr.get_seg_model(config)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    # print(model)
    
    input_names = ['image']
    output_names = ['output1', 'output2']
    dynamic_axes = {'image': {0: '-1'}, 'output1': {0: '-1'}, 'output2': {0: '-1'}}
    dummy_input = torch.randn(1, 3, 1024, 2048)
    torch.onnx.export(model, dummy_input, "hrnet.onnx",
                     input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes,
                     opset_version=11, verbose=True)

if __name__ == "__main__":
    convert()