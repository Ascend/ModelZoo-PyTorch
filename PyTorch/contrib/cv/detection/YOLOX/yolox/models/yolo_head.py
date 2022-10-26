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


#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import math
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolox.utils import bboxes_iou

from .losses import IOUloss
from .network_blocks import BaseConv, DWConv


class YOLOXHead(nn.Module):
    def __init__(
        self,
        num_classes,
        width=1.0,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False,
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__()

        self.n_anchors = 1
        self.num_classes = num_classes
        self.decode_in_inference = True  # for deploy, set to False

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        Conv = DWConv if depthwise else BaseConv

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * 1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, xin, labels=None, imgs=None):
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            if self.training:
                output = torch.cat([reg_output.float(), obj_output.float(), cls_output.float()], 1)
                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level, xin[0].dtype, xin[0].device
                )
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(
                    torch.zeros(1, grid.shape[1])
                    .fill_(stride_this_level)
                    .type_as(xin[0])
                )
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(
                        batch_size, self.n_anchors, 4, hsize, wsize
                    )
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 4
                    )
                    origin_preds.append(reg_output.clone())

            else:
                output = torch.cat(
                    [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
                )

            outputs.append(output)

        if self.training:
            return self.get_losses(
                imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                torch.cat(outputs, 1),
                origin_preds,
                dtype=xin[0].dtype,
            )
        else:
            self.hw = [x.shape[-2:] for x in outputs]
            # [batch, n_anchors_all, 85]
            outputs = torch.cat(
                [x.flatten(start_dim=2) for x in outputs], dim=2
            ).permute(0, 2, 1)
            if self.decode_in_inference:
                return self.decode_outputs(outputs, dtype=xin[0].dtype, device=xin[0].device)
            else:
                return outputs

    def get_output_and_grid(self, output, k, stride, dtype, device):
        grid = self.grids[k].float()

        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).to(dtype).to(device).float()
            self.grids[k] = grid

        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, self.n_anchors * hsize * wsize, -1
        )
        grid = grid.view(1, -1, 2)
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        return output, grid

    def decode_outputs(self, outputs, dtype, device):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).to(dtype).to(device)
        strides = torch.cat(strides, dim=1).to(dtype).to(device)

        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs

    def get_losses(
        self,
        imgs,
        x_shifts,
        y_shifts,
        expanded_strides,
        labels,
        outputs,
        origin_preds,
        dtype,
    ):
        bbox_preds = outputs[:, :, :4].contiguous()  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4].unsqueeze(-1).contiguous()  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:].contiguous()  # [batch, n_anchors_all, n_cls]

        # calculate targets
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1).float()
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []
        fg_masks_inboxes = []

        # Prepare some constant tensors for higher performance,
        # while operations between tensor and scalar not performing very well on NPU.
        const_tensor_0 = torch.tensor(0.0, device=imgs.device)
        const_tensor_1 = torch.tensor(1.0, device=imgs.device)
        const_tensor_2 = torch.tensor(2.0, device=imgs.device)
        const_tensor_5 = torch.tensor(0.5, device=imgs.device)
        const_tensor_25 = torch.tensor(2.5, device=imgs.device)

        const_tensor_dict = {}
        const_tensor_dict['0'] = const_tensor_0
        const_tensor_dict['1'] = const_tensor_1
        const_tensor_dict['2'] = const_tensor_2
        const_tensor_dict['0.5'] = const_tensor_5
        const_tensor_dict['2.5'] = const_tensor_25

        num_fg = const_tensor_0
        num_gts = 0.0

        nlabel_list = nlabel.tolist()
        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel_list[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                num_gt_pad = min(int((num_gt + 32 - 1) // 32 * 32), labels.shape[1])

                gt_bboxes_per_image = labels[batch_idx, :num_gt_pad, 1:5]  # [num_gt_pad, 4]
                gt_classes = labels[batch_idx, :num_gt_pad, 0]  # [num_gt_pad, 4]
                bboxes_preds_per_image = bbox_preds[batch_idx]  # [n_anchors_all, 4]

                try:
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                        fg_mask_inboxes,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        num_gt_pad,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                        const_tensor_dict,
                    )
                except RuntimeError:
                    logger.error(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    torch.npu.empty_cache()
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                        fg_mask_inboxes
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        num_gt_pad,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                        const_tensor_dict,
                        "cpu",
                    )

                num_fg += num_fg_img

                cls_target = torch.npu_one_hot(
                    gt_matched_classes.to(torch.int32), -1, self.num_classes, 1, 0
                ) * pred_ious_this_matching.unsqueeze(-1)  # [num_gt_pad, self.num_classes]
                obj_target = fg_mask.unsqueeze(-1)  # [n_anchors_all, 1]
                reg_target = gt_bboxes_per_image.index_select(0, matched_gt_inds)  # [num_gt_pad, 4]
                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((fg_mask.shape[0], 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0],
                        x_shifts=x_shifts[0],
                        y_shifts=y_shifts[0],
                    )

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            fg_masks_inboxes.append(fg_mask_inboxes)
            if self.use_l1:
                l1_targets.append(l1_target)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        fg_masks_inboxes = torch.cat(fg_masks_inboxes, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        num_fg = max(num_fg, 1)
        num_fg_pad = int((num_fg + 128 - 1) // 128 * 128)

        mask = torch.zeros(num_fg_pad)
        mask[:int(num_fg)] = 1.
        mask = mask.npu()

        reg_targets_pad = torch.zeros(num_fg_pad, 4)
        reg_targets_pad[:int(num_fg), :] = reg_targets[fg_masks_inboxes.bool()].float().cpu()
        reg_targets_pad = reg_targets_pad.npu()

        bbox_preds_pad = torch.zeros(num_fg_pad, 4)
        bbox_preds_pad[:int(num_fg), :] = bbox_preds.view(-1, 4)[fg_masks_inboxes.bool()].float().cpu()
        bbox_preds_pad = bbox_preds_pad.npu()

        loss_iou = (
            self.iou_loss(bbox_preds_pad.float(), reg_targets_pad.float()) * mask
        ).sum() / num_fg
        loss_obj = (
            self.bcewithlog_loss(obj_preds.view(-1, 1).float(), obj_targets.float())  # backward overflow
        ).sum() / num_fg
        loss_cls = (
            self.bcewithlog_loss(
                cls_preds.view(-1, self.num_classes).float(), cls_targets.float()
            ) * fg_masks_inboxes.unsqueeze(-1)
        ).sum() / num_fg
        if self.use_l1:
            loss_l1 = (
                self.l1_loss(origin_preds.view(-1, 4), l1_targets) * fg_masks_inboxes.unsqueeze(-1)
            ).sum() / num_fg
        else:
            loss_l1 = 0.0

        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1

        return (
            loss,
            reg_weight * loss_iou,
            loss_obj,
            loss_cls,
            loss_l1,
            num_fg / max(num_gts, 1),
        )

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    @torch.no_grad()
    def get_assignments(
        self,
        batch_idx,
        num_gt,
        num_gt_pad,
        total_num_anchors,
        gt_bboxes_per_image,
        gt_classes,
        bboxes_preds_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        cls_preds,
        bbox_preds,
        obj_preds,
        labels,
        imgs,
        const_tensor_dict,
        mode="gpu",
    ):

        if mode == "cpu":
            print("------------CPU Mode for This Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        mask_gt_pad = None
        if num_gt_pad > num_gt:
            mask_gt_pad = torch.zeros(num_gt_pad, 1)
            mask_gt_pad[:num_gt] = 1.
            mask_gt_pad = mask_gt_pad.to(gt_bboxes_per_image.device)

        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
            num_gt_pad,
            const_tensor_dict,
            mask_gt_pad,
        )

        bboxes_preds_per_image = bboxes_preds_per_image  # [n_anchors_all, 4]
        cls_preds_ = cls_preds[batch_idx]
        obj_preds_ = obj_preds[batch_idx]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        pair_wise_ious = bboxes_iou(gt_bboxes_per_image.float().clone(), bboxes_preds_per_image.float().clone(), False)

        gt_cls_per_image = (
            torch.npu_one_hot(gt_classes.to(torch.int32), -1, self.num_classes, 1, 0)
            .float()
            .unsqueeze(1)
            .repeat(1, num_in_boxes_anchor, 1)
        )
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        cls_preds_ = (
            cls_preds_.float().unsqueeze(0).repeat(num_gt_pad, 1, 1).sigmoid_()
            * obj_preds_.float().unsqueeze(0).repeat(num_gt_pad, 1, 1).sigmoid_()
        )
        pair_wise_cls_loss = F.binary_cross_entropy(
            cls_preds_.float().sqrt_(), gt_cls_per_image.float(), reduction="none"
        ).sum(-1)
        del cls_preds_

        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + 100000.0 * (1. - is_in_boxes_and_center)
        )

        mask = fg_mask.unsqueeze(0).repeat(num_gt_pad, 1).float()
        if num_gt_pad > num_gt:
            mask = mask * mask_gt_pad

        cost = cost * mask + 1000000.0 * (1. - mask)

        pair_wise_ious = pair_wise_ious * mask

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
            fg_mask_inboxes,
        ) = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask, const_tensor_dict)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.npu()
            fg_mask = fg_mask.npu()
            pred_ious_this_matching = pred_ious_this_matching.npu()
            matched_gt_inds = matched_gt_inds.npu()

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
            fg_mask_inboxes,
        )

    def get_in_boxes_info(
        self,
        gt_bboxes_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        total_num_anchors,
        num_gt,
        num_gt_pad,
        const_tensor_dict,
        mask_gt_pad,
    ):
        expanded_strides_per_image = expanded_strides[0]
        x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
        y_shifts_per_image = y_shifts[0] * expanded_strides_per_image
        x_centers_per_image = (
            (x_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt_pad, 1)
        )  # [n_anchor] -> [n_gt, n_anchor]
        y_centers_per_image = (
            (y_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt_pad, 1)
        )

        gt_bboxes_per_image_trans = gt_bboxes_per_image.transpose(0, 1).contiguous()
        gt_bboxes_0 = gt_bboxes_per_image_trans[0]
        gt_bboxes_1 = gt_bboxes_per_image_trans[1]
        gt_bboxes_2 = gt_bboxes_per_image_trans[2]
        gt_bboxes_3 = gt_bboxes_per_image_trans[3]

        gt_bboxes_per_image_l = (
            (gt_bboxes_0 - 0.5 * gt_bboxes_2)
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_r = (
            (gt_bboxes_0 + 0.5 * gt_bboxes_2)
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_t = (
            (gt_bboxes_1 - 0.5 * gt_bboxes_3)
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_b = (
            (gt_bboxes_1 + 0.5 * gt_bboxes_3)
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )

        b1 = torch.stack([x_centers_per_image, gt_bboxes_per_image_r, y_centers_per_image, gt_bboxes_per_image_b], 0)
        b2 = torch.stack([gt_bboxes_per_image_l, x_centers_per_image, gt_bboxes_per_image_t, y_centers_per_image], 0)
        bbox_deltas = b1 - b2

        is_in_boxes = bbox_deltas.min(dim=0).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0
        # in fixed center

        center_radius = 2.5

        gt_bboxes_0_anchors = gt_bboxes_0.unsqueeze(1).repeat(1, total_num_anchors).contiguous()
        gt_bboxes_1_anchors = gt_bboxes_1.unsqueeze(1).repeat(1, total_num_anchors).contiguous()
        center_radius_strided = (center_radius * expanded_strides_per_image.unsqueeze(0)).contiguous()

        gt_bboxes_per_image_l = gt_bboxes_0_anchors - center_radius_strided
        gt_bboxes_per_image_r = gt_bboxes_0_anchors + center_radius_strided
        gt_bboxes_per_image_t = gt_bboxes_1_anchors - center_radius_strided
        gt_bboxes_per_image_b = gt_bboxes_1_anchors + center_radius_strided

        c1 = torch.stack([x_centers_per_image, gt_bboxes_per_image_r, y_centers_per_image, gt_bboxes_per_image_b], 0)
        c2 = torch.stack([gt_bboxes_per_image_l, x_centers_per_image, gt_bboxes_per_image_t, y_centers_per_image], 0)
        center_deltas = c1 - c2
        
        is_in_centers = center_deltas.min(dim=0).values > 0.0
        if num_gt_pad > num_gt:
            is_in_centers = is_in_centers * mask_gt_pad
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # in boxes and in centers
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

        is_in_boxes_and_center = (
            is_in_boxes.float() * is_in_centers
        )
        return is_in_boxes_anchor.float(), is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask, const_tensor_dict):
        # Dynamic K
        # ---------------------------------------------------------------
        matching_matrix = torch.zeros_like(cost)

        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
        topk_ious, _ = torch.topk(ious_in_boxes_matrix.half(), n_candidate_k, dim=1)  # fp16 to use aicore TopKD
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)

        if num_gt > 2:
            device = matching_matrix.device
            min_k = torch.min(dynamic_ks)
            max_k = torch.max(dynamic_ks)
            min_k, max_k = min_k.item(), max_k.item()
            if min_k == max_k:
                _, pos_idxes = torch.topk(-cost, k=max_k, dim=1, largest=True)
                matching_matrix.scatter_(1, pos_idxes, 1)
            else:
                offsets = torch.arange(0, matching_matrix.shape[0] * matching_matrix.shape[1],
                        step=matching_matrix.shape[1], dtype=torch.int, device=device)[:, None]
                masks = (torch.arange(0, max_k, dtype=dynamic_ks.dtype,
                    device=device)[None, :].expand(matching_matrix.shape[0], max_k) < dynamic_ks[:, None]) # 120 is num_gt
                _, pos_idxes = torch.topk(-cost, k=max_k, dim=1, largest=True)
                pos_idxes.add_(offsets)
                pos_idxes = torch.masked_select(pos_idxes, masks)
                matching_matrix.view(-1)[pos_idxes] = 1
        else:
            ks = dynamic_ks.tolist()
            for gt_idx in range(num_gt):
                _, pos_idx = torch.topk(-cost[gt_idx], k=ks[gt_idx], largest=True)
                matching_matrix[gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks

        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > const_tensor_dict['0']:
            mask = anchor_matching_gt > 1
            _, cost_argmin = torch.min(cost, dim=0)
            cost_argmin = torch.masked_select(cost_argmin, mask)
            matching_matrix = matching_matrix * (~mask)
            matching_matrix[cost_argmin, mask] = 1.0

        fg_mask_inboxes = matching_matrix.sum(0) > 0.
        fg_mask_inboxes = fg_mask_inboxes.float()
        num_fg = (fg_mask_inboxes * fg_mask).sum()

        fg_mask_clone = fg_mask.clone()
        fg_mask_inboxes = fg_mask_inboxes * fg_mask_clone
        fg_mask1 = fg_mask * (1. - fg_mask_clone) + fg_mask_inboxes * fg_mask_clone
        fg_mask.copy_(fg_mask1)

        matched_gt_inds = matching_matrix.argmax(0)
        gt_matched_classes = gt_classes.index_select(0, matched_gt_inds)

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)

        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds, fg_mask_inboxes
