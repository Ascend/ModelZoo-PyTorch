#!/bin/bash
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

# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import coco as cfg
from ..box_utils import match, log_sum_exp, refine_match

def sep_smoothl1loss(x1, x2):
    l1 = torch.nn.functional.l1_loss(x1, x2, reduce=False)
    l2 = ((x1 - x2) ** 2) * 0.5
    return torch.where(l1 < 1, l2, l1 - 0.5).sum()

class RefineDetMultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True, theta=0.01, use_ARM=False, npu_device=None):
        super(RefineDetMultiBoxLoss, self).__init__()
        self.npu_device = npu_device
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = cfg['variance']
        self.theta = theta
        self.use_ARM = use_ARM

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            targets (list of tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        arm_loc_data, arm_conf_data, odm_loc_data, odm_conf_data, priors = predictions
        

        if self.use_ARM:
            loc_data, conf_data = odm_loc_data, odm_conf_data
        else:
            loc_data, conf_data = arm_loc_data, arm_conf_data

        num = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))
        num_classes = self.num_classes
        #print(loc_data.size(), conf_data.size(), priors.size())

        # match priors (default boxes) and ground truth boxes

        ###################  modify npu begin  #################################
        loc_t = torch.Tensor(num, num_priors, 4).to(self.npu_device)
        ######################### modify npu end  ###################################

        conf_t = torch.LongTensor(num, num_priors)
        MAX_LEN = 40
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data

            ## 动态shape固定 begin
            num_gt = truths.size(0)
            pad_bbox = torch.zeros(size=(MAX_LEN, 4), dtype=truths.dtype, device=truths.device)
            pad_labels = torch.zeros(size=(MAX_LEN, ), dtype=labels.dtype, device=labels.device)
            pad_bbox[:num_gt] = truths
            pad_labels[:num_gt] = labels
            truths = pad_bbox
            labels = pad_labels
            ## 动态shape固定 end

            if num_classes == 2:
                '''
                print("="*50)
                print('labels\t', labels)
                print('labels size\t', labels.size())
                print('labels dtpye\t', labels.dtype)
                print('labels device\t', labels.device)
                print("=" * 50)
                '''
                labels = labels >= 0
                '''
                print('labels\t', labels)
                print("=" * 50)
                print('labels size\t', labels.size())
                print('labels dtpye\t', labels.dtype)
                print('labels device\t', labels.device)
                print("=" * 50)
                '''
            defaults = priors.data.to(self.npu_device)
            if self.use_ARM:
                refine_match(self.threshold, truths, defaults, 
                    self.variance, labels, loc_t, conf_t, idx, arm_loc_data[idx].data, num_gt=num_gt)
            else:
                refine_match(self.threshold, truths, defaults, 
                    self.variance, labels, loc_t, conf_t, idx, num_gt=num_gt)
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        elif self.npu_device:
            loc_t = loc_t.to(self.npu_device)
            conf_t = conf_t.to(self.npu_device)
        # wrap targets
        #loc_t = Variable(loc_t, requires_grad=False)
        #conf_t = Variable(conf_t, requires_grad=False)
        loc_t.requires_grad = False
        conf_t.requires_grad = False
        #print(loc_t.size(), conf_t.size())
        if self.use_ARM:
            P = F.softmax(arm_conf_data, 2)
            arm_conf_tmp = P[:,:,1]
            object_score_index = arm_conf_tmp <= self.theta
            pos = conf_t > 0
            pos[object_score_index.data] = 0
        else:
            pos = conf_t > 0

        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)

        ## 固定 动态shape begin
        pad_loc_p = (loc_data * pos_idx).view(-1, 4)
        pad_loc_t = (loc_t * pos_idx).view(-1, 4)
        # loss_l = F.smooth_l1_loss(pad_loc_p, pad_loc_t, reduction='sum')
        loss_l = sep_smoothl1loss(pad_loc_p, pad_loc_t)
        
        ## 固定 动态shape end

        # loc_p = loc_data[pos_idx].view(-1, 4)  ## 动态shape
        # loc_t = loc_t[pos_idx].view(-1, 4)     ## 动态shape
        # loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')
        # print(loss_l, pad_loss_l)
        
        '''
        print('-' * 20)
        print('loss_l:\t ',loss_l)
        print('loss_l shape\t', loss_l.shape)
        print('loss_l dtype\t', loss_l.dtype)
        print('loss_l device\t', loss_l.device)
        print('=' * 20)
        '''
        ###########

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
        
        loss_c[pos.view(-1,1)] = 0  # filter out pos boxes for now

        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)
        #print(num_pos.size(), num_neg.size(), neg.size())

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)

        
        mask = (pos+neg).gt(0).flatten()
        pad_conf_p = (conf_data * (pos_idx+neg_idx).gt(0)).view(-1, self.num_classes)
        pad_targets_weighted = (conf_t * (pos+neg).gt(0)).flatten()  # (pos+neg).gt(0) torch.bool
        pad_loss_c = F.cross_entropy(pad_conf_p, pad_targets_weighted, reduction='none')
        loss_c = (pad_loss_c * mask).sum()

        # conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        # targets_weighted = conf_t[(pos + neg).gt(0)]
        # loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')
        

        N = num_pos.data.sum().float()
        #N = max(num_pos.data.sum().float(), 1)
        loss_l /= N
        loss_c /= N
        #print(N, loss_l, loss_c)

        return loss_l, loss_c
