# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# Copyright 2020 Huawei Technologies Co., Ltd
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
from utils import utils
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class CrossEntropyCriterion(_Loss):

    def __init__(self, args):
        super().__init__()
        self.padding_idx = args.padding_idx

    def forward(self, norm_probs, target, reduce=True):
        """Compute the loss for the given sample.
        """
        lprobs = norm_probs.view(-1, norm_probs.size(-1))
        target = target.view(-1)
        loss = F.nll_loss(lprobs, target, size_average=False, ignore_index=self.padding_idx,
                          reduce=reduce)
        return loss


class LabelSmoothedCrossEntropyCriterion(_Loss):

    def __init__(self, args):
        super().__init__()
        self.eps = args.label_smoothing
        self.padding_idx = args.padding_idx
        # loc = 'npu:{}'.format(self.args.device_id)
        # self.device = loc

    def forward(self, norm_probs, target, reduce=True):
        """Compute the loss for the given sample.
        """
        target = target.view(-1, 1)
        lprobs = norm_probs.view(-1, norm_probs.size(-1))
        non_pad_mask = target.ne(self.padding_idx)
        nll_loss = -lprobs.gather(dim=-1, index=target.cpu().long().npu())
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)

        nll_loss = nll_loss.masked_fill(~non_pad_mask, 0).sum()
        smooth_loss = smooth_loss.masked_fill(~non_pad_mask, 0).sum()

        eps_i = self.eps / lprobs.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss
        return loss


CRITERION_REGISTRY = {
    'label_smoothed_cross_entropy': LabelSmoothedCrossEntropyCriterion,
    'cross_entropy': CrossEntropyCriterion
}
