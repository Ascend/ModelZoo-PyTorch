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
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def iiou(pred, target, size_average = True):

    combination = target[:,:,:,:]*pred[:,:,:,:]
    sumlist_combination = torch.sum(combination,[-1,2]).npu() #compute SG
    sumlist_pred = torch.sum(pred,[-1,2]).npu()  # compute S
    sumlist_target = torch.sum(target,[-1,2]).npu()  # compute G
    #print('compression matrix',sumlist_target)


    iou0 = 1-(sumlist_combination[0]/(sumlist_pred[0]+sumlist_target[0]-sumlist_combination[0]))
    iou1 = 1-(sumlist_combination[1]/(sumlist_pred[1]+sumlist_target[1]-sumlist_combination[1]))
    iou2 = 1-(sumlist_combination[2]/(sumlist_pred[2]+sumlist_target[2]-sumlist_combination[2]))
    iou3 = 1-(sumlist_combination[3]/(sumlist_pred[3]+sumlist_target[3]-sumlist_combination[3]))
    iou4 = 1-(sumlist_combination[4]/(sumlist_pred[4]+sumlist_target[4]-sumlist_combination[4]))
    iou5 = 1-(sumlist_combination[5]/(sumlist_pred[5]+sumlist_target[5]-sumlist_combination[5]))
    iou6 = 1-(sumlist_combination[6]/(sumlist_pred[6]+sumlist_target[6]-sumlist_combination[6]))
    iou7 = 1-(sumlist_combination[7]/(sumlist_pred[7]+sumlist_target[7]-sumlist_combination[7]))
    
    IoU = (iou0+iou1+iou2+iou3+iou4+iou5+iou6+iou7)/8

    
    
    #b = pred.shape[0]
    #i=0
    #IoU = 0.0


    #Iand1 = torch.sum(target[:,:,:,:]*pred[:,:,:,:])


    #Ior1 = torch.sum(target[:,:,:,:]) + torch.sum(pred[:,:,:,:])-Iand1
        
        
    #IoU1 = Iand1/Ior1

        

    #IoU loss is (1-IoU1)
    #IoU = IoU + (1-IoU1)

    return IoU


