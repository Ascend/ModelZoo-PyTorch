# Copyright 2023 Huawei Technologies Co., Ltd
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
import torch.nn as nn
import torch.nn.functional as F

from mmseg.structures.sampler import OHEMPixelSampler


def sample(self, seg_logit, seg_label):
    """Sample pixels that have high loss or with low prediction confidence.

    Args:
        seg_logit (torch.Tensor): segmentation logits, shape (N, C, H, W)
        seg_label (torch.Tensor): segmentation label, shape (N, 1, H, W)

    Returns:
        torch.Tensor: segmentation weight, shape (N, H, W)
    """
    with torch.no_grad():
        assert seg_logit.shape[2:] == seg_label.shape[2:]
        assert seg_label.shape[1] == 1
        seg_label = seg_label.squeeze(1).long()
        batch_kept = self.min_kept * seg_label.size(0)
        valid_mask = seg_label != self.context.ignore_index
        valid_mask = torch.nonzero(valid_mask, as_tuple=True)
        seg_weight = seg_logit.new_zeros(size=seg_label.size())
        valid_seg_weight = seg_weight[valid_mask]
        if self.thresh is not None:
            seg_prob = F.softmax(seg_logit, dim=1)

            tmp_seg_label = seg_label.clone().unsqueeze(1)
            tmp_seg_label[tmp_seg_label == self.context.ignore_index] = 0
            seg_prob = seg_prob.gather(1, tmp_seg_label).squeeze(1)
            sort_prob, sort_indices = seg_prob[valid_mask].sort()

            if sort_prob.numel() > 0:
                min_threshold = sort_prob[min(batch_kept,
                                              sort_prob.numel() - 1)]
            else:
                min_threshold = 0.0
            threshold = max(min_threshold, self.thresh)
            valid_seg_weight[seg_prob[valid_mask] < threshold] = 1.
        else:
            if not isinstance(self.context.loss_decode, nn.ModuleList):
                losses_decode = [self.context.loss_decode]
            else:
                losses_decode = self.context.loss_decode
            losses = 0.0
            for loss_module in losses_decode:
                losses += loss_module(
                    seg_logit,
                    seg_label,
                    weight=None,
                    ignore_index=self.context.ignore_index,
                    reduction_override='none')

            # faster than topk according to https://github.com/pytorch/pytorch/issues/22812  # noqa
            _, sort_indices = losses[valid_mask].sort(descending=True)
            valid_seg_weight[sort_indices[:batch_kept]] = 1.

        seg_weight[valid_mask] = valid_seg_weight

        return seg_weight


OHEMPixelSampler.sample = sample
