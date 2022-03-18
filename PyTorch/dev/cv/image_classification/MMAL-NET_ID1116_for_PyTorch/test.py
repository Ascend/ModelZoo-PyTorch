#coding=utf-8
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
import torch
import torch.nn as nn
import sys
from tqdm import tqdm
from config import input_size, root, proposalN, channels
from util.read_dataset import read_dataset
from util.auto_laod_resume import auto_load_resume
from networks.model import MainNet

import os
import torch.npu
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
	NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
	torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

CUDA = torch.npu.is_available()
DEVICE = torch.device(f'npu:{NPU_CALCULATE_DEVICE}')

# dataset
set = 'CUB'
if set == 'CUB':
    root = './datasets/CUB_200_2011'  # dataset path
    # model path
    pth_path = "./models/cub_epoch144.pth"
    num_classes = 200
elif set == 'Aircraft':
    root = './datasets/FGVC-aircraft'  # dataset path
    # model path
    pth_path = "./models/air_epoch146.pth"
    num_classes = 100

batch_size = 10

#load dataset
_, testloader = read_dataset(input_size, batch_size, root, set)

# 瀹氫箟妯″瀷
model = MainNet(proposalN=proposalN, num_classes=num_classes, channels=channels)

model = model.to(f'npu:{NPU_CALCULATE_DEVICE}')
criterion = nn.CrossEntropyLoss()

#鍔犺浇checkpoint
if os.path.exists(pth_path):
    epoch = auto_load_resume(model, pth_path, status='test')
else:
    sys.exit('There is not a pth exist.')

print('Testing')
raw_correct = 0
object_correct = 0
model.eval()
with torch.no_grad():
    for i, data in enumerate(tqdm(testloader)):
        if set == 'CUB':
            x, y, boxes, _ = data
        else:
            x, y = data
        x = x.to(f'npu:{NPU_CALCULATE_DEVICE}')
        y = y.to(f'npu:{NPU_CALCULATE_DEVICE}')
        local_logits, local_imgs = model(x, epoch, i, 'test', DEVICE)[-2:]
        # local
        pred = local_logits.max(1, keepdim=True)[1]
        object_correct += pred.eq(y.view_as(pred)).sum().item()

    print('\nObject branch accuracy: {}/{} ({:.2f}%)\n'.format(
            object_correct, len(testloader.dataset), 100. * object_correct / len(testloader.dataset)))