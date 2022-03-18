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

import os
import sys
sys.path.append("pix2pixHD")
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch
import torch.onnx


def pth2onnx(output_file):
    model = create_model(opt).netG
    model.eval()
    input_names = ["input_concat"]
    output_names = ["fake_image"]
    dynamic_axes = {'input_concat': {0: '-1'}, 'fake_image': {0: '-1'}}
    dummy_input = torch.randn(1, 36, 1024, 2048)
    torch.onnx.export(model, dummy_input, output_file, input_names = input_names, \
    dynamic_axes = dynamic_axes, output_names = output_names, verbose=True, opset_version=11)


if __name__ == "__main__":
    opt = TestOptions().parse(save=False)
    opt.name = "label2city_1024p"
    opt.netG = "local"
    opt.ngf = 32
    opt.resize_or_crop = "none"
    pth2onnx(opt.output_file)



