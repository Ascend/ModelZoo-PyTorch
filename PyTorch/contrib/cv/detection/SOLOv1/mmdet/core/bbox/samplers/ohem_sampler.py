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

import torch

from ..transforms import bbox2roi
from .base_sampler import BaseSampler


class OHEMSampler(BaseSampler):
    """
    Online Hard Example Mining Sampler described in [1]_.

    References:
        .. [1] https://arxiv.org/pdf/1604.03540.pdf
    """

    def __init__(self,
                 num,
                 pos_fraction,
                 context,
                 neg_pos_ub=-1,
                 add_gt_as_proposals=True,
                 **kwargs):
        super(OHEMSampler, self).__init__(num, pos_fraction, neg_pos_ub,
                                          add_gt_as_proposals)
        if not hasattr(context, 'num_stages'):
            self.bbox_roi_extractor = context.bbox_roi_extractor
            self.bbox_head = context.bbox_head
        else:
            self.bbox_roi_extractor = context.bbox_roi_extractor[
                context.current_stage]
            self.bbox_head = context.bbox_head[context.current_stage]

    def hard_mining(self, inds, num_expected, bboxes, labels, feats):
        with torch.no_grad():
            rois = bbox2roi([bboxes])
            bbox_feats = self.bbox_roi_extractor(
                feats[:self.bbox_roi_extractor.num_inputs], rois)
            cls_score, _ = self.bbox_head(bbox_feats)
            loss = self.bbox_head.loss(
                cls_score=cls_score,
                bbox_pred=None,
                labels=labels,
                label_weights=cls_score.new_ones(cls_score.size(0)),
                bbox_targets=None,
                bbox_weights=None,
                reduction_override='none')['loss_cls']
            _, topk_loss_inds = loss.topk(num_expected)
        return inds[topk_loss_inds]

    def _sample_pos(self,
                    assign_result,
                    num_expected,
                    bboxes=None,
                    feats=None,
                    **kwargs):
        # Sample some hard positive samples
        pos_inds = torch.nonzero(assign_result.gt_inds > 0)
        if pos_inds.numel() != 0:
            pos_inds = pos_inds.squeeze(1)
        if pos_inds.numel() <= num_expected:
            return pos_inds
        else:
            return self.hard_mining(pos_inds, num_expected, bboxes[pos_inds],
                                    assign_result.labels[pos_inds], feats)

    def _sample_neg(self,
                    assign_result,
                    num_expected,
                    bboxes=None,
                    feats=None,
                    **kwargs):
        # Sample some hard negative samples
        neg_inds = torch.nonzero(assign_result.gt_inds == 0)
        if neg_inds.numel() != 0:
            neg_inds = neg_inds.squeeze(1)
        if len(neg_inds) <= num_expected:
            return neg_inds
        else:
            return self.hard_mining(neg_inds, num_expected, bboxes[neg_inds],
                                    assign_result.labels[neg_inds], feats)
