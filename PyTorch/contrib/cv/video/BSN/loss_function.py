# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn.functional as F
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')
    
def bi_loss(scores,anchors,opt):
    scores = scores.view(-1).npu()
    anchors = anchors.contiguous().view(-1)
    # 0916: cuda -> npu 
    pmask = (scores>opt["tem_match_thres"]).float().npu()
    num_positive = torch.sum(pmask)
    num_entries = len(scores)
    ratio=num_entries/num_positive

    coef_0=0.5*(ratio)/(ratio-1)
    coef_1=coef_0*(ratio-1)
    loss = coef_1*pmask*torch.log(anchors+0.00001) + coef_0*(1.0-pmask)*torch.log(1.0-anchors+0.00001)
    loss=-torch.mean(loss)
    num_sample=[torch.sum(pmask),ratio] 
    return loss,num_sample

def TEM_loss_calc(anchors_action,anchors_start,anchors_end,
             match_scores_action,match_scores_start,match_scores_end,opt):
    
    loss_action,num_sample_action=bi_loss(match_scores_action,anchors_action,opt)
    loss_start_small,num_sample_start_small=bi_loss(match_scores_start,anchors_start,opt)
    loss_end_small,num_sample_end_small=bi_loss(match_scores_end,anchors_end,opt)

    loss_dict={"loss_action":loss_action,"num_sample_action":num_sample_action,
          "loss_start":loss_start_small,"num_sample_start":num_sample_start_small,
          "loss_end":loss_end_small,"num_sample_end":num_sample_end_small}
    #print loss_dict
    return loss_dict


def TEM_loss_function(y_action,y_start,y_end,TEM_output,opt):
    anchors_action = TEM_output[:,0,:]
    anchors_start = TEM_output[:,1,:]
    anchors_end = TEM_output[:,2,:]
    loss_dict=TEM_loss_calc(anchors_action,anchors_start,anchors_end,
                     y_action,y_start,y_end,opt)
    
    cost=2*loss_dict["loss_action"]+loss_dict["loss_start"]+loss_dict["loss_end"]
    loss_dict["cost"] = cost
    return loss_dict

def PEM_loss_function(anchors_iou,match_iou,model,opt):
    match_iou = match_iou.npu()
    anchors_iou = anchors_iou.view(-1)
    u_hmask = (match_iou>opt["pem_high_iou_thres"]).float()
    u_mmask = ((match_iou<=opt["pem_high_iou_thres"]) & (match_iou>opt["pem_low_iou_thres"])).float()
    u_lmask = (match_iou<opt["pem_low_iou_thres"]).float()

    num_h=torch.sum(u_hmask)
    num_m=torch.sum(u_mmask)
    num_l=torch.sum(u_lmask)

    r_m= model.module.u_ratio_m * num_h/(num_m)
    r_m = torch.min(r_m, torch.Tensor([1.0]).npu())[0]
    u_smmask = torch.Tensor(np.random.rand(u_hmask.size()[0])).npu()
    u_smmask=u_smmask*u_mmask
    u_smmask = (u_smmask > (1. -r_m)).float()
    
    r_l= model.module.u_ratio_l * num_h/(num_l)
    r_l=torch.min(r_l, torch.Tensor([1.0]).npu())[0]
    u_slmask = torch.Tensor(np.random.rand(u_hmask.size()[0])).npu()
    u_slmask=u_slmask*u_lmask
    u_slmask=(u_slmask > (1. -r_l)).float()
    
    iou_weights=u_hmask+u_smmask+u_slmask
    iou_loss = F.smooth_l1_loss(anchors_iou,match_iou)
    iou_loss = torch.sum(iou_loss * iou_weights) / torch.sum(iou_weights)
    
    return iou_loss

