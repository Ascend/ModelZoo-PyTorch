#
# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================
#
"""
Created on Sat Mar 6 2021

@author: Kuan-Lin Chen
"""
import torch
import argparse
from data import data_dict
from models import model_dict
from test import test
from utils import dir_path,cuda_device
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')

parser = argparse.ArgumentParser(description='Empirical study on ResNets, ResNEsts, and A-ResNEsts',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data', default='CIFAR10', choices=['CIFAR10','CIFAR100'], help='image classification datasets')
parser.add_argument('--checkpoint_folder',default='./checkpoint/', type=dir_path, help='path to the checkpoint folder')
parser.add_argument('--device', default='npu:0', type=cuda_device, help='specify a CUDA device')
parser.add_argument('--seed_list', default=[0,1,2,3,4,5,6], nargs='+', type=int, help='train a model with different random seeds')
parser.add_argument('--model', default='CIFAR10_Standard_ResNet_110', choices=list(model_dict.keys()), help='the DNN model')
args = parser.parse_args()

torch.backends.cudnn.deterministic = True
print(torch.npu.get_device_name(int(args.device[-1]))) # print the GPU

testset = data_dict[args.data]().testset

for seed in args.seed_list:
    name = args.model+'_seed='+str(seed)
    print('start testing the model '+name)
    net = model_dict[args.model]()
    pretrained_model = torch.load(args.checkpoint_folder+name+'/last_model.pt')
    net.load_state_dict(pretrained_model,strict=True)
    net = net.to(f'npu:{NPU_CALCULATE_DEVICE}')
 
    test(net,testset,args.device,args.checkpoint_folder+name)
