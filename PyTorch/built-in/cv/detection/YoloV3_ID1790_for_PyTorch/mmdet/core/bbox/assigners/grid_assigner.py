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

from ..builder import BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


@BBOX_ASSIGNERS.register_module()
class GridAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `-1`, `0`, or a positive integer
    indicating the ground truth index.

    - -1: don't care
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        pos_iou_thr (float): IoU threshold for positive bboxes.
        neg_iou_thr (float or tuple): IoU threshold for negative bboxes.
        min_pos_iou (float): Minimum iou for a bbox to be considered as a
            positive bbox. Positive samples can have smaller IoU than
            pos_iou_thr due to the 4th step (assign max IoU sample to each gt).
        gt_max_assign_all (bool): Whether to assign all bboxes with the same
            highest overlap with some gt to that gt.
    """
    g_zero = None
    g_neg_one = None

    def __init__(self,
                 pos_iou_thr,
                 neg_iou_thr,
                 min_pos_iou=.0,
                 gt_max_assign_all=True,
                 iou_calculator=dict(type='BboxOverlaps2D')):
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou
        self.gt_max_assign_all = gt_max_assign_all
        self.iou_calculator = build_iou_calculator(iou_calculator)

    def assign(self, bboxes, box_responsible_flags, gt_bboxes, gt_labels=None):
        """Assign gt to bboxes. The process is very much like the max iou
        assigner, except that positive samples are constrained within the cell
        that the gt boxes fell in.

        This method assign a gt bbox to every bbox (proposal/anchor), each bbox
        will be assigned with -1, 0, or a positive number. -1 means don't care,
        0 means negative sample, positive number is the index (1-based) of
        assigned gt.
        The assignment is done in following steps, the order matters.

        1. assign every bbox to -1
        2. assign proposals whose iou with all gts <= neg_iou_thr to 0
        3. for each bbox within a cell, if the iou with its nearest gt >
            pos_iou_thr and the center of that gt falls inside the cell,
            assign it to that bbox
        4. for each gt bbox, assign its nearest proposals within the cell the
            gt bbox falls in to itself.

        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            box_responsible_flags (Tensor): flag to indicate whether box is
                responsible for prediction, shape(n, )
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        num_gts, num_bboxes = gt_bboxes.size(0), bboxes.size(0)

        # compute iou between all gt and bboxes
        # overlaps = self.iou_calculator(gt_bboxes, bboxes)
        overlaps = torch_npu.npu_ptiou(bboxes, gt_bboxes)


        # 1. assign -1 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.float)

        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes, ))
            if num_gts == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes, ),
                                                    -1,
                                                    dtype=torch.float)
            return AssignResult(
                num_gts,
                assigned_gt_inds,
                max_overlaps,
                labels=assigned_labels)

        # 2. assign negative: below
        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        # shape of max_overlaps == argmax_overlaps == num_bboxes

        # torch_npu.npu_max returns index as int32
        max_overlaps, argmax_overlaps = torch_npu.npu_max(overlaps, dim=0)

        if isinstance(self.neg_iou_thr, float):
            # assigned_gt_inds[(max_overlaps >= 0)
            #                  & (max_overlaps <= self.neg_iou_thr)] = 0
            if GridAssigner.g_zero is None:
                GridAssigner.g_zero = torch.zeros_like(assigned_gt_inds)
            assert GridAssigner.g_zero.shape == assigned_gt_inds.shape
            assigned_gt_inds = torch.where((max_overlaps >= 0) &
                                           (max_overlaps <= self.neg_iou_thr),
                                           GridAssigner.g_zero, assigned_gt_inds)
        elif isinstance(self.neg_iou_thr, (tuple, list)):
            assert len(self.neg_iou_thr) == 2
            assigned_gt_inds[(max_overlaps > self.neg_iou_thr[0])
                             & (max_overlaps <= self.neg_iou_thr[1])] = 0

        # 3. assign positive: falls into responsible cell and above
        # positive IOU threshold, the order matters.
        # the prior condition of comparision is to filter out all
        # unrelated anchors, i.e. not box_responsible_flags
        # overlaps[:, ~box_responsible_flags.type(torch.bool)] = -1.

        # cast unint8 to bool through int32, avoiding calling aicpu 'Cast',
        # while aicore 'Cast' not supporting unint8 to bool
        flag = box_responsible_flags.int().type(torch.bool)
        flag = flag[None, :]
        if GridAssigner.g_neg_one is None:
            GridAssigner.g_neg_one = -1. * torch.ones_like(overlaps)
        assert GridAssigner.g_neg_one.shape == overlaps.shape
        overlaps = torch.where(flag, overlaps, GridAssigner.g_neg_one)


        # calculate max_overlaps again, but this time we only consider IOUs
        # for anchors responsible for prediction
        max_overlaps, argmax_overlaps = torch_npu.npu_max(overlaps, dim=0)

        # for each gt, which anchor best overlaps with it
        # for each gt, the max iou of all proposals
        # shape of gt_max_overlaps == gt_argmax_overlaps == num_gts
        gt_max_overlaps, gt_argmax_overlaps = torch_npu.npu_max(overlaps, dim=1)

        assigned_gt_inds = torch_npu.npu_grid_assign_positive(
            assigned_gt_inds, overlaps, box_responsible_flags,
            max_overlaps, argmax_overlaps, gt_max_overlaps, gt_argmax_overlaps,
            num_gts, self.pos_iou_thr, self.min_pos_iou, self.gt_max_assign_all)

        # pos_inds = (max_overlaps >
        #             self.pos_iou_thr) & box_responsible_flags.type(torch.bool)
        # assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1
        #
        # # 4. assign positive to max overlapped anchors within responsible cell
        # for i in range(num_gts):
        #     if gt_max_overlaps[i] > self.min_pos_iou:
        #         if self.gt_max_assign_all:
        #             max_iou_inds = (overlaps[i, :] == gt_max_overlaps[i]) & \
        #                  box_responsible_flags.type(torch.bool)
        #             assigned_gt_inds[max_iou_inds] = i + 1
        #         elif box_responsible_flags[gt_argmax_overlaps[i]]:
        #             assigned_gt_inds[gt_argmax_overlaps[i]] = i + 1

        # assign labels of positive anchors
        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1.0)
            # pos_inds = torch.nonzero(
            #     assigned_gt_inds > 0, as_tuple=False).squeeze()
            # if pos_inds.numel() > 0:
            #     assigned_labels[pos_inds] = gt_labels[
            #         assigned_gt_inds[pos_inds] - 1]
            pos_inds = (assigned_gt_inds > 0).float()
            neg_pos_inds = 1 - pos_inds

            if pos_inds.sum() > 0:
                agi_pos = (assigned_gt_inds - 1) * pos_inds
                assigned_labels = assigned_labels * neg_pos_inds +\
                                  gt_labels.index_select(0, agi_pos.int()) * pos_inds

        else:
            assigned_labels = None

        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)
