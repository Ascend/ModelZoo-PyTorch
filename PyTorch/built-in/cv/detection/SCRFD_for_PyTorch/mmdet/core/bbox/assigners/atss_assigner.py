# Copyright (c) 2020 Huawei Technologies Co., Ltd
# All rights reserved.
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

import torch
if torch.__version__ >= "1.8":
    import torch_npu
from ..builder import BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


@BBOX_ASSIGNERS.register_module()
class ATSSAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `0` or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        topk (float): number of bbox selected in each level
    """

    def __init__(self,
                 topk,
                 mode=0,
                 iou_calculator=dict(type='BboxOverlaps2D'),
                 ignore_iof_thr=-1):
        self.topk = topk
        self.mode = mode
        self.iou_calculator = build_iou_calculator(iou_calculator)
        self.ignore_iof_thr = ignore_iof_thr

        self.INF = 100000000
        MAXLEN_LIST = [8, 16, 32, 64, 128, 1024, 2048]
        self.range_dict = {}
        self.offset_dict = {}
        self.inf_dict = {}
        self.overlaps_inf_dict = {}
        self.index_dict = {}
        self.src_dict = {}
        self.gt_buffer = torch.tensor([0.001], dtype=torch.float32).npu()
        self.num_bboxes = 16800
        self.topk_num = self.topk*3
        for t in MAXLEN_LIST:
            self.range_dict[t] = torch.arange(t).npu()
            self.offset_dict[t] = torch.arange(0, t*self.num_bboxes, step = self.num_bboxes).repeat(self.topk_num).view(-1, t).npu()
            self.inf_dict[t] = torch.full((t, self.num_bboxes), -self.INF, dtype=torch.float32).npu()
            self.overlaps_inf_dict[t] = torch.full((t, self.num_bboxes), -self.INF, dtype=torch.float32).view(-1).npu()
            self.index_dict[t] = torch.zeros(self.num_bboxes*t, dtype=torch.float32).npu()
            self.src_dict[t] = torch.ones(self.topk_num*t,dtype=torch.float32).npu()
        
    # https://github.com/sfzhang15/ATSS/blob/master/atss_core/modeling/rpn/atss/loss.py

    def assign(self,
               bboxes,
               num_level_bboxes,
               gt_bboxes,
               gt_bboxes_ignore=None,
               gt_labels=None,
               real_gt=128):
        """Assign gt to bboxes.

        The assignment is done in following steps

        1. compute iou between all bbox (bbox of all pyramid levels) and gt
        2. compute center distance between all bbox and gt
        3. on each pyramid level, for each gt, select k bbox whose center
           are closest to the gt center, so we total select k*l bbox as
           candidates for each gt
        4. get corresponding iou for the these candidates, and compute the
           mean and std, set mean + std as the iou threshold
        5. select these candidates whose iou are greater than or equal to
           the threshold as postive
        6. limit the positive sample's center in gt


        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            num_level_bboxes (List): num of bboxes in each level
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        INF = self.INF
        num_bboxes = bboxes.size(0)
        num_gt = real_gt
        MAX_LEN = gt_bboxes.size(0)
        #print('AT1:', num_gt, num_bboxes)

        # compute iou between all bbox and gt
        if torch.__version__ >= "1.8":
            overlaps = torch_npu.contrib.function.npu_iou(bboxes, gt_bboxes, mode='ptiou')
        else:
            overlaps = self.iou_calculator(bboxes, gt_bboxes)

        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes, ))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes, ),
                                                    -1,
                                                    dtype=torch.long)
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        # compute center distance between all bbox and gt
        gt_bboxes_t = gt_bboxes.t().contiguous()
        bboxes_t = bboxes.t().contiguous()
        gt_cx = (gt_bboxes_t[0, :] + gt_bboxes_t[2, :]) / 2.0
        gt_cy = (gt_bboxes_t[1, :] + gt_bboxes_t[3, :]) / 2.0

        gt_width = gt_bboxes_t[2, :] - gt_bboxes_t[0, :]
        gt_height = gt_bboxes_t[3, :] - gt_bboxes_t[1, :]
        gt_area = torch.sqrt( torch.clamp(gt_width*gt_height, min=1e-4) )

        bboxes_cx = (bboxes_t[0, :] + bboxes_t[2, :]) / 2.0
        bboxes_cy = (bboxes_t[1, :] + bboxes_t[3, :]) / 2.0
        distance_cx = (bboxes_cx[:, None, None] - gt_cx[None, :, None]).squeeze().pow(2)
        distance_cy = (bboxes_cy[:, None, None] - gt_cy[None, :, None]).squeeze().pow(2)
        distances = (distance_cx + distance_cy).sqrt()

        if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                and gt_bboxes_ignore.numel() > 0 and bboxes.numel() > 0):
            assert RuntimeError('gt_bboxes_ignore is not None')
            ignore_overlaps = self.iou_calculator(
                bboxes, gt_bboxes_ignore, mode='iof')
            ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            ignore_idxs = ignore_max_overlaps > self.ignore_iof_thr
            distances[ignore_idxs, :] = INF
            assigned_gt_inds[ignore_idxs] = -1
        # Selecting candidates based on the center distance
        candidate_idxs = []
        start_idx = 0
        for level, bboxes_per_level in enumerate(num_level_bboxes):
            # on each pyramid level, for each gt,
            # select k bbox whose center are closest to the gt center
            end_idx = start_idx + bboxes_per_level
            distances_per_level = distances[start_idx:end_idx, :] #(A,G)
            selectable_k = min(self.topk, bboxes_per_level)
            _, topk_idxs_per_level = distances_per_level.topk(
                selectable_k, dim=0, largest=False)
            #print('AT-LEVEL:', start_idx, end_idx, bboxes_per_level, topk_idxs_per_level.shape)
            candidate_idxs.append(topk_idxs_per_level + start_idx)
            start_idx = end_idx
        candidate_idxs = torch.cat(candidate_idxs, dim=0)# candidate anchors (topk*num_level_bboxes, G) = (AK, G)

        # get corresponding iou for the these candidates, and compute the
        # mean and std, set mean + std as the iou threshold
        maxlen_arange = self.range_dict[MAX_LEN]
        candidate_overlaps = overlaps[candidate_idxs, maxlen_arange] #(AK,G)

        overlaps_mean_per_gt = candidate_overlaps.mean(0)
        overlaps_std_per_gt = candidate_overlaps.std(0)
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt

        is_pos = candidate_overlaps >= overlaps_thr_per_gt[None, :]
        #print('CAND:', candidate_idxs.shape, candidate_overlaps.shape, is_pos.shape)
        #print('BOXES:', bboxes_cx.shape)

        # limit the positive sample's center in gt
        offset_arange = self.offset_dict[MAX_LEN]
        candidate_idxs_src = candidate_idxs
        candidate_idxs = candidate_idxs + offset_arange

        ep_bboxes_cx_process = bboxes_cx.index_select(0,candidate_idxs_src.view(-1)).view(-1,MAX_LEN)
        ep_bboxes_cy_process = bboxes_cy.index_select(0,candidate_idxs_src.view(-1)).view(-1,MAX_LEN)
        candidate_idxs = candidate_idxs.view(-1)

        # calculate the left, top, right, bottom distance between positive
        # bbox center and gt side
        l_ = ep_bboxes_cx_process - gt_bboxes_t[0, :]
        t_ = ep_bboxes_cy_process - gt_bboxes_t[1, :]
        r_ = gt_bboxes_t[2, :] - ep_bboxes_cx_process
        b_ = gt_bboxes_t[3, :] - ep_bboxes_cy_process
        #is_in_gts = torch.stack([l_, t_, r_, b_], dim=1).min(dim=1)[0] > 0.01
        dist_min = torch.stack([l_, t_, r_, b_], dim=1).min(dim=1)[0] # (A,G)
        dist_min.div_(gt_area)
        #print('ATTT:', l_.shape, t_.shape, dist_min.shape, self.mode)
        if self.mode==0:
            is_in_gts = dist_min > self.gt_buffer
        elif self.mode==1:
            is_in_gts = dist_min > -0.25
        elif self.mode==2:
            is_in_gts = dist_min > -0.15
            #dist_expand = torch.clamp(gt_area / 16.0, min=1.0, max=3.0)
            #dist_min.mul_(dist_expand)
            #is_in_gts = dist_min > -0.25
        elif self.mode==3:
            dist_expand = torch.clamp(gt_area / 16.0, min=1.0, max=6.0)
            dist_min.mul_(dist_expand)
            is_in_gts = dist_min > -0.2
        elif self.mode==4:
            dist_expand = torch.clamp(gt_area / 16.0, min=0.5, max=6.0)
            dist_min.mul_(dist_expand)
            is_in_gts = dist_min > -0.2
        elif self.mode==5:
            dist_div = torch.clamp(gt_area / 16.0, min=0.5, max=3.0)
            dist_min.div_(dist_div)
            is_in_gts = dist_min > -0.2
        else:
            raise ValueError
        #print(gt_area.shape, is_in_gts.shape, is_pos.shape)
        is_pos = is_pos & is_in_gts
        # if an anchor box is assigned to multiple gts,
        # the one with the highest IoU will be selected.
        overlaps_inf = self.overlaps_inf_dict[MAX_LEN]
        overlaps_t = overlaps.t().contiguous().view(-1)
        index = self.index_dict[MAX_LEN]
        src = self.src_dict[MAX_LEN] * is_pos.view(-1)
        index_ = index.scatter(dim=0,index=candidate_idxs,src=src)
        overlaps_inf = torch.where(index_>0, overlaps_t, overlaps_inf)
        overlaps_inf = overlaps_inf.view(MAX_LEN, -1)
        mask_ = (self.range_dict[MAX_LEN]<num_gt)[:,None].repeat(1,num_bboxes)
        overlaps_inf = torch.where(mask_>0, overlaps_inf, self.inf_dict[MAX_LEN])
        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=0)
        max_mask = (max_overlaps != -INF)
        assigned_gt_inds = (argmax_overlaps + 1) * max_mask

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
            pos_inds = (assigned_gt_inds <= 0)
            if pos_inds.numel() > 0:
                assigned_labels = assigned_labels * pos_inds
        else:
            raise RuntimeError('gt_labels should not be None')
        return AssignResult(
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
