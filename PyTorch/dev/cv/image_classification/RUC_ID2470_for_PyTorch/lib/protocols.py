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
from lib.utils import AverageMeter
import time
from sklearn.utils.linear_assignment_ import linear_assignment
import numpy as np
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')

def _hungarian_match(flat_preds, flat_targets, num_samples, class_num):  
    num_k = class_num
    num_correct = np.zeros((num_k, num_k))
  
    for c1 in range(0, num_k):
        for c2 in range(0, num_k):
        # elementwise, so each sample contributes once
            votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
            num_correct[c1, c2] = votes
  
    # num_correct is small
    match = linear_assignment(num_samples - num_correct)
  
    # return as list of tuples, out_c to gt_c
    res = []
    for out_c, gt_c in match:
        res.append((out_c, gt_c))
  
    return res


def test(net, testloader,device, class_num):
    net.eval()
    predicted_all = []
    targets_all = []
    for batch_idx, (inputs, _,_, targets, indexes) in enumerate(testloader):
        batchSize = inputs.size(0)
        targets, inputs = targets.to(f'npu:{NPU_CALCULATE_DEVICE}'), inputs.to(f'npu:{NPU_CALCULATE_DEVICE}')
        with torch.no_grad():
            output = net(inputs)
        predicted = torch.argmax(output, 1)
        predicted_all.append(predicted)
        targets_all.append(targets)
    

    flat_predict = torch.cat(predicted_all).to(f'npu:{NPU_CALCULATE_DEVICE}')
    flat_target = torch.cat(targets_all).to(f'npu:{NPU_CALCULATE_DEVICE}')
    num_samples = flat_predict.shape[0]
    match = _hungarian_match(flat_predict, flat_target, num_samples, class_num)
    reordered_preds = torch.zeros(num_samples).to(f'npu:{NPU_CALCULATE_DEVICE}')
    
    for pred_i, target_i in match:
        reordered_preds[flat_predict == pred_i] = int(target_i)
        
    acc = int((reordered_preds == flat_target.float()).sum()) / float(num_samples) * 100
        
    return acc, reordered_preds

def test_ruc(net, net2, testloader, device, class_num):
    net.eval()
    net2.eval()
    
    predicted_all = [[] for i in range(0,3)]
    targets_all = []
    acc_list = []
    p_label_list = []
    
    for batch_idx, (inputs, _, _, targets, indexes) in enumerate(testloader):
        batchSize = inputs.size(0)
        targets, inputs = targets.to(f'npu:{NPU_CALCULATE_DEVICE}'), inputs.to(f'npu:{NPU_CALCULATE_DEVICE}')
        logit = net(inputs)
        logit2 = net2(inputs)
        _, predicted = torch.max(logit, 1)
        _, predicted2 = torch.max(logit2, 1)
        _, predicted3 = torch.max(logit + logit2, 1)
        
        predicted_all[0].append(predicted)
        predicted_all[1].append(predicted2)
        predicted_all[2].append(predicted3)
        targets_all.append(targets)
    
    for i in range(0, 3):
        flat_predict = torch.cat(predicted_all[i]).to(f'npu:{NPU_CALCULATE_DEVICE}')
        flat_target = torch.cat(targets_all).to(f'npu:{NPU_CALCULATE_DEVICE}')
        num_samples = flat_predict.shape[0]
        acc = int((flat_predict.float() == flat_target.float()).sum()) / float(num_samples) * 100
        acc_list.append(acc)
        p_label_list.append(flat_predict)
        
    return acc_list, p_label_list