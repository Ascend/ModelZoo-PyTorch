# -*- coding: utf-8 -*-
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


from datetime import datetime
import numpy as np
import argparse
import os

from model.initialization import initialization
from model.utils import evaluation

import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model.network import TripletLoss, SetNet
from model.utils import TripletSampler


parser = argparse.ArgumentParser()
parser.add_argument('--input_path', default='', type=str,
                    help='input_path: input path of checkpoint file for onnx and om conversion'
                         ' before pth2onnx.py and to om. Default: \'\'')
parser.add_argument('--iters', default=-1, type=int,
                    help='iters: loaded iteration of input model'
                         ' Default: -1, load from config')

class wrapperNet(nn.Module):
    def __init__(self, module):
        super(wrapperNet, self).__init__()
        self.module = module

def load(load_path, model, restore_iter):
    loaded = torch.load(load_path, map_location=torch.device('cpu'))
    model.load_state_dict(loaded)

def convert(input_path, output_path, restore_iter, hidden_dim):
    align_size = 100
    
    print(f'Init the model of iteration {restore_iter}...')
    encoder = SetNet(hidden_dim).float()
    encoder = encoder
    encoder = wrapperNet(encoder)
    
    print(f'Loading the model of iteration {restore_iter}...')
    load(input_path, encoder, restore_iter)
    print('Model loaded.')
    
    encoder.eval()
    input_names = ["image_seq"]
    output_names = ["feature"]
	
    dummy_input = torch.randn((1, align_size, 64, 44))
    
    print('Exporting model to onnx...')
    torch.onnx.export(encoder.module, dummy_input, output_path, input_names = input_names, output_names = output_names, opset_version=11, verbose=False)
    print('Onnx export done.')


if __name__ == "__main__":
    args = parser.parse_args()
    from config_1p import conf_1p
    from config_8p import conf_8p
    
    work_abspath = osp.abspath(conf_8p['WORK_PATH'])
    conf_model = conf_8p['model']
    conf_data = conf_8p['data']
    
    model = conf_model['model_name']
    dataset = conf_data['dataset']
    pid = conf_data['pid_num']
    shuffle = 'True' if conf_data['pid_shuffle'] else 'False'
    dim = conf_model['hidden_dim']
    margin = str(conf_model['margin'])
    bs = conf_model['batch_size'][0] * conf_model['batch_size'][-1]
    hoft = conf_model['hard_or_full_trip']
    frame = conf_model['frame_num']
    iters = conf_model['total_iter']
    if args.iters != -1:
        iters = args.iters
        
    pth_prefix = f'{model}_{dataset}_{pid}_{shuffle}_{dim}_{margin}_{bs}_{hoft}_{frame}-{iters}'
    onnx_path = osp.join(work_abspath, '../gaitset_submit.onnx')
    pth_path = osp.join(work_abspath, f'checkpoint/{model}/{pth_prefix}-encoder.ptm')
    
    if args.input_path != '':
        pth_path = args.input_path
    
    convert(pth_path, onnx_path, iters, dim)
