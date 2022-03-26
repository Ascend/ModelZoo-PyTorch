# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import random
import shutil
import time
import warnings
import math
import glob
import numpy as np
import sys

import torch
import torch.npu
import torch.nn as nn
from collections import OrderedDict
import torch.onnx
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../'))
import mnasnet

# modelarts modification
import moxing as mox


CACHE_TRAINING_URL = "/cache/training"
CACHE_MODEL_URL = "/cache/model"

def proc_node_module(checkpoint, AttrName):
    new_state_dict = OrderedDict()
    for k, v in checkpoint[AttrName].items():
        if k[0:7] == "module.":
            name = k[7:]
        else:
            name = k[0:]
        new_state_dict[name] = v
    return new_state_dict


def convert(pth_file, onnx_path, class_num, train_url):

    checkpoint = torch.load(pth_file, map_location=None)
    
    model = mnasnet.mnasnet1_0(num_classes=class_num)
    model.load_state_dict(checkpoint)

    model.eval()
    
    input_names = ["image"]
    output_names = ["class"]
    dynamic_axes = {'image': {0: '-1'}, 'class': {0: '-1'}}
    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(model, dummy_input, onnx_path, input_names=input_names, output_names=output_names,  opset_version=11)
    mox.file.copy_parallel(onnx_path, train_url + 'model.onnx')

def convert_pth_to_onnx(config_args):
    mox.file.copy_parallel(config_args.is_best_name, os.path.join(CACHE_MODEL_URL, "checkpoint.pth.tar"))
    pth_pattern = os.path.join(CACHE_MODEL_URL, 'checkpoint.pth.tar')
    pth_file_list = glob.glob(pth_pattern)
    if not pth_file_list:
        print(f"can't find pth {pth_pattern}")
        return
    pth_file = pth_file_list[0]
    onnx_path = pth_file.split(".")[0] + '.onnx'
    convert(pth_file, onnx_path, config_args.class_num, config_args.train_url)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    # modelarts
    parser.add_argument('--data_url', metavar='DIR', default='/cache/data_url', help='path to dataset')
    parser.add_argument('--train_url', default="/cache/training",
                        type=str,
                        help="setting dir of training output")
    parser.add_argument('--onnx', default=True, action='store_true',
                        help="convert pth model to onnx")
    parser.add_argument('--class_num', default=1000, type=int,
                        help='number of class')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='mnasnet1_0')
    parser.add_argument('--is_best_name', dest='is_best_name',
                        help=' weight dir')
    args = parser.parse_args()
    print('===========================')
    print(args)
    print('===========================')
    convert_pth_to_onnx(args)
