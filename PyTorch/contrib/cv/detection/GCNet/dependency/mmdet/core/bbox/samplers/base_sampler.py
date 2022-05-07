# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
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
#

from abc import ABCMeta, abstractmethod

import torch

from .sampling_result import SamplingResult


class BaseSampler(metaclass=ABCMeta):
    """Base class of samplers."""

    def __init__(self,
                 num,
                 pos_fraction,
                 neg_pos_ub=-1,
                 add_gt_as_proposals=True,
                 **kwargs):
        self.num = num
        self.pos_fraction = pos_fraction
        self.neg_pos_ub = neg_pos_ub
        self.add_gt_as_proposals = add_gt_as_proposals
        self.pos_sampler = self
        self.neg_sampler = self

    @abstractmethod
    def _sample_pos(self, assign_result, num_expected, **kwargs):
        """Sample positive samples."""
        pass

    @abstractmethod
    def _sample_neg(self, assign_result, num_expected, **kwargs):
        """Sample negative samples."""
        pass

    def sample(self,
               assign_result,
               bboxes,
               gt_bboxes,
               gt_labels=None,
               **kwargs):
        """Sample positive and negative bboxes.

        This is a simple implementation of bbox sampling given candidates,
        assigning results and ground truth bboxes.

        Args:
            assign_result (:obj:`AssignResult`): Bbox assigning results.
            bboxes (Tensor): Boxes to be sampled from.
            gt_bboxes (Tensor): Ground truth bboxes.
            gt_labels (Tensor, optional): Class labels of ground truth bboxes.

        Returns:
            :obj:`SamplingResult`: Sampling result.

        Example:
            >>> from mmdet.core.bbox import RandomSampler
            >>> from mmdet.core.bbox import AssignResult
            >>> from mmdet.core.bbox.demodata import ensure_rng, random_boxes
            >>> rng = ensure_rng(None)
            >>> assign_result = AssignResult.random(rng=rng)
            >>> bboxes = random_boxes(assign_result.num_preds, rng=rng)
            >>> gt_bboxes = random_boxes(assign_result.num_gts, rng=rng)
            >>> gt_labels = None
            >>> self = RandomSampler(num=32, pos_fraction=0.5, neg_pos_ub=-1,
            >>>                      add_gt_as_proposals=False)
            >>> self = self.sample(assign_result, bboxes, gt_bboxes, gt_labels)
        """
#         print(torch.npu.synchronize(),'==================sample attr')

        static_gt_size = 40
        gt_nums = gt_bboxes.size(0)
        if len(bboxes.shape) < 2:
            bboxes = bboxes[None, :]
        
        bboxes = bboxes[:, :4]
#         gt_bboxes_static = gt_bboxes.new_zeros((static_gt_size,4))
#         gt_bboxes_static[:gt_bboxes.size(0)] = gt_bboxes
#         gt_bboxes = gt_bboxes_static
        
#         print(torch.npu.synchronize(),'==================A1')
        gt_flags = bboxes.new_zeros((bboxes.shape[0], ), dtype=torch.uint8)
        if self.add_gt_as_proposals and len(gt_bboxes) > 0:
#             print('========add gt:', assign_result.gt_inds.size())
            if gt_labels is None:
                raise ValueError(
                    'gt_labels must be given when add_gt_as_proposals is True')
#             print('gt labels:',gt_labels)
#             nopad_gt_num = torch.nonzero(gt_labels < 80, as_tuple=False).numel()
            nopad_gt_num = (gt_labels < 80).sum()
            nopad_gt = (gt_labels < 80)
#             print('nopad_gt_num1:',nopad_gt_num)
            bboxes = torch.cat([gt_bboxes, bboxes], dim=0)
            assign_result.add_gt_(gt_labels)
#             print(torch.npu.synchronize(),'==================A2')
#             gt_ones = bboxes.new_ones(gt_bboxes.shape[0], dtype=torch.uint8)
            gt_ones = bboxes.new_zeros(gt_bboxes.shape[0], dtype=torch.uint8)
#             gt_ones[:nopad_gt_num] = 1
            gt_ones = gt_ones + nopad_gt.byte()
            gt_flags = torch.cat([gt_ones, gt_flags])
            
#         print(torch.npu.synchronize(),'==================A3')
        
        num_expected_pos = int(self.num * self.pos_fraction)
#         print(torch.npu.synchronize(),'--------self sampler:',num_expected_pos,self.num,self.pos_fraction)
        pos_inds = self.pos_sampler._sample_pos(
            assign_result, num_expected_pos, bboxes=bboxes, **kwargs)
        
        # We found that sampled indices have duplicated items occasionally.
        # (may be a bug of PyTorch)
#         pos_inds = pos_inds.unique()
#         num_sampled_pos = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False).numel()
        num_sampled_pos = (assign_result.gt_inds > 0).sum()
        num_expected_neg = self.num - num_sampled_pos
        if self.neg_pos_ub >= 0:
#             print(torch.npu.synchronize(),'==================A5.1')
            _pos = max(1, num_sampled_pos)
            neg_upper_bound = int(self.neg_pos_ub * _pos)
            if num_expected_neg > neg_upper_bound:
                num_expected_neg = neg_upper_bound
                
#         print(torch.npu.synchronize(),'==================A5.2:',self.neg_sampler._sample_neg)
        neg_inds = self.neg_sampler._sample_neg(
            assign_result, num_expected_neg, bboxes=bboxes, **kwargs)
#         neg_inds = neg_inds.unique()
#         print(torch.npu.synchronize(),'==================A6')
        sampling_result = SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes,
                                         assign_result, gt_flags)
#         print(torch.npu.synchronize(),'==================A7')
        return sampling_result
