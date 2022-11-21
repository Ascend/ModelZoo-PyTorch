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

from __future__ import print_function
import sys
sys.path.append('./DG-Net')
from trainer import DGNet_Trainer, to_gray
from utils import get_config
import argparse
from torch.autograd import Variable
import torchvision.utils as vutils
import sys
import torch
import random
import os
import numpy as np
from torchvision import datasets, models, transforms
from PIL import Image
from shutil import copyfile
from DGnet import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, default="E0.5new_reid0.5_w30000", help="model name")
    parser.add_argument('--batchsize', default=1, type=int, help='batchsize')
    parser.add_argument('--which_epoch', default=100000, type=int, help='iteration')
    parser.add_argument('--output', default='DG-net.onnx', type=str, help='save file')



    opts = parser.parse_args()
    opts.checkpoint_gen = "./outputs/%s/checkpoints/gen_00%06d.pt"%(opts.name, opts.which_epoch)
    opts.checkpoint_id = "./outputs/%s/checkpoints/id_00%06d.pt"%(opts.name, opts.which_epoch)
    opts.config = './outputs/%s/config.yaml'%opts.name
    config = get_config(opts.config)
    model = DGNet_test(config)
    state_dict_gen = torch.load(opts.checkpoint_gen, map_location='cpu')
    model.gen_a.load_state_dict(state_dict_gen['a'], strict=False)
    model.gen_b = model.gen_a
    model.gen_a.eval()
    model.gen_b.eval()
    state_dict_id = torch.load(opts.checkpoint_id, map_location='cpu')
    model.id_a.load_state_dict(state_dict_id['a'])
    model.id_b = model.id_a
    model.id_a.eval()
    model.id_b.eval()
    model.cpu()
    model.eval()
    input_names = ["image1", 'image2']
    output_names = ["output"]
    input1 = torch.randn(opts.batchsize, 1, 256, 128)
    input2 = torch.randn(opts.batchsize, 3, 256, 128)
    torch.onnx.export(model, (input1, input2), opts.output, input_names=input_names,
                    output_names = output_names, opset_version=10, verbose=True)