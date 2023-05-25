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

import torch
if torch.__version__ >= '1.8':
    import torch_npu
import torch.nn as nn

from main_utils import BoxCoder


class IntegratedLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2.0):
        super(IntegratedLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.box_coder = BoxCoder()

    def forward(self, classifications, regressions, anchors, refined_achors, annotations,
                md_thres=0.5, mining_param=(1, 0., -1), pos_max_num=1024):
        classifications = classifications.float()
        regressions = regressions.float()
        train_device = classifications.device
        batch_size = classifications.shape[0]
        boxes_dim = 5
        gt_boxes_all = annotations.shape[1]
        alpha, beta, var = mining_param

        range_index = torch.arange(0, gt_boxes_all, device=train_device).long()

        gt_boxes = annotations[:, :, :-1]
        gt_label = annotations[:, :, -1]
        gt_bool = (gt_label != -1)
        gt_boxes_num = gt_bool.sum(1)

        sa = torch_npu.npu_rotated_iou(anchors, gt_boxes, True, 0, True)
        fa = torch_npu.npu_rotated_iou(refined_achors, gt_boxes, True, 0, True)
        md = torch.abs((alpha * sa + beta * fa) - torch.abs(fa - sa) ** var)

        iou_max, iou_argmax = md.max(2)
        positive_indices = torch.ge(iou_max, md_thres)
        max_gt, argmax_gt = md.max(1)
        argmax_gt_pos = ((max_gt < md_thres) & gt_bool)
        # import ipdb;ipdb.set_trace(context = 15)
        for batch_index in range(batch_size):
            positive_indices[batch_index, argmax_gt[batch_index, argmax_gt_pos[batch_index]]] = True
        positive_anchors_num = positive_indices.sum(1).clamp(min=0, max=pos_max_num).long()
        positive_anchors_bool = torch.zeros((batch_size, pos_max_num), device=train_device).bool()
        for batch_index in range(batch_size):
            positive_anchors_bool[batch_index, :positive_anchors_num[batch_index]] = True
        # matching-weight
        pos_shape = (batch_size, pos_max_num, gt_boxes_all)
        pos = torch.zeros(pos_shape, dtype=md.dtype, device=train_device)
        pos[positive_anchors_bool] = md[positive_indices]
        pos_mask = torch.ge(pos, md_thres)
        max_pos, armmax_pos = pos.max(1)
        for batch_index in range(batch_size):
            bboxes_num = gt_boxes_num[batch_index]
            pos_mask[batch_index, armmax_pos[batch_index, :bboxes_num], range_index[:bboxes_num]] = 1
        comp = torch.where(pos_mask, (1 - max_pos).unsqueeze(1).repeat(1, pos_max_num, 1), pos)
        matching_weight = comp + pos
        match_weight_data, match_weight_index = matching_weight.max(2)
        # cls loss
        classifications_clamp = torch.clamp(classifications, 1e-4, 1.0 - 1e-4)
        cls_targets = torch.ones(classifications_clamp.shape, device=train_device) * -1
        cls_targets[torch.lt(iou_max, md_thres - 0.1)] = 0
        cls_targets[positive_indices] = 0
        assigned_gt_label = torch.gather(gt_label, 1, iou_argmax)
        pos_assigned_gt_label = assigned_gt_label[positive_indices].long()
        cls_targets[positive_indices, pos_assigned_gt_label] = 1

        alpha_factor = torch.ones(cls_targets.shape, device=train_device) * self.alpha
        alpha_factor = torch.where(torch.eq(cls_targets, 1.), alpha_factor, 1. - alpha_factor)
        focal_weight = torch.where(torch.eq(cls_targets, 1.), 1. - classifications_clamp, classifications_clamp)
        focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)
        bin_cross_entropy = -(cls_targets * torch.log(classifications_clamp + 1e-6) + (1.0 - cls_targets) * torch.log(
            1.0 - classifications_clamp + 1e-6))

        soft_weight = torch.zeros(classifications_clamp.shape, device=train_device)
        soft_weight = torch.where(torch.eq(cls_targets, 0.), torch.ones_like(cls_targets), soft_weight)

        matching_weight_data_1 = match_weight_data + 1
        soft_weight[positive_indices, pos_assigned_gt_label] = matching_weight_data_1[positive_anchors_bool]
        cls_loss = focal_weight * bin_cross_entropy * soft_weight

        cls_loss = torch.where(torch.ne(cls_targets, -1.0), cls_loss, torch.zeros(cls_loss.shape, device=train_device))
        loss_cls = (cls_loss.sum((1, 2)) / torch.clamp(positive_anchors_num, min=1.0)).mean(dim=0, keepdim=True)

        # reg loss
        reg_shape = (batch_size, pos_max_num, boxes_dim)
        pos_regression = torch.zeros(reg_shape, dtype=regressions.dtype, device=train_device)
        pos_all_rois = torch.zeros(reg_shape, dtype=anchors.dtype, device=train_device)
        pos_assigned_gt_boxes = torch.zeros(reg_shape, dtype=gt_boxes.dtype, device=train_device)
        assigned_gt_boxes_list = []
        for batch_index in range(batch_size):
            assigned_gt_boxes_batch = gt_boxes[batch_index, iou_argmax[batch_index]]
            assigned_gt_boxes_list.append(assigned_gt_boxes_batch)
        assigned_gt_boxes = torch.stack(assigned_gt_boxes_list, dim=0)
        pos_regression[positive_anchors_bool] = regressions[positive_indices]
        pos_all_rois[positive_anchors_bool] = anchors[positive_indices]
        pos_assigned_gt_boxes[positive_anchors_bool] = assigned_gt_boxes[positive_indices]
        reg_targets = self.box_coder.batch_encode(pos_all_rois, pos_assigned_gt_boxes)
        loss_reg = batch_smooth_l1_loss(pos_regression, reg_targets, match_weight_data, positive_anchors_num)
        return loss_cls, loss_reg


def batch_smooth_l1_loss(inputs,
                         targets,
                         weight,
                         positive_num,
                         beta=1. / 9):
    """
    https://github.com/facebookresearch/maskrcnn-benchmark
    """
    diff = torch.abs(inputs - targets)
    loss = torch.where(
        diff < beta,
        0.5 * diff ** 2 / beta,
        diff - 0.5 * beta
    ) * weight.unsqueeze(2)
    return (loss.sum((1, 2)) / (positive_num * 5)).mean(dim=0, keepdim=True)
