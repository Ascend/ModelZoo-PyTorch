from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
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
CENTERNET_PATH = './CenterNet/src'
sys.path.insert(0, CENTERNET_PATH)
MODEL_PATH = '../models/ctdet_coco_dla_2x.pth'
import os
import _init_paths
import torch
import torch.utils.data
from opts import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset_factory import get_dataset


def convert():
    device = torch.device("cpu")
    torch.set_default_tensor_type(torch.FloatTensor)
    # device = torch.device("cuda")
    # torch.set_default_tensor_type(torch.cuda.FloatTensor)
    TASK = 'ctdet' 
    opt = opts().parse('{} --load_model {}'.format(TASK, MODEL_PATH).split(' '))
    Dataset = get_dataset(opt.dataset, opt.task)
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    model = create_model(opt.arch, opt.heads, opt.head_conv) 
    model = load_model(model, input_file, None, opt.resume, opt.lr, opt.lr_step)
    model.eval()
    
    input_names = ["actual_input"]
    output_names = ["output1","output2","output3"]
    dynamic_axes = {'actual_input': {0: '-1'}, 'output1': {0: '-1'}, 'output2': {0: '-1'}, 'output3': {0: '-1'}}
    dummy_input = torch.randn(1, 3, 512, 512)
    torch.onnx.export(model, dummy_input, output_file, input_names = input_names, dynamic_axes = dynamic_axes, output_names = output_names, opset_version=11, verbose=True)

if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    convert()
