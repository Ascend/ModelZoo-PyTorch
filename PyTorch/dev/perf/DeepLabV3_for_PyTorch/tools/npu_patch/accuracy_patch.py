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

from typing import Union, Tuple, List

import torch
import torch.nn as nn
from torch import Tensor

from mmseg.utils import SampleList
from mmseg.models.utils import resize

from mmseg.models.decode_heads.decode_head import BaseDecodeHead


def accuracy(pred, target, topk=1, thresh=None, ignore_index=None):
    """Calculate accuracy according to the prediction and target.

    Args:
        pred (torch.Tensor): The model prediction, shape (N, num_class, ...)
        target (torch.Tensor): The target of each prediction, shape (N, , ...)
        ignore_index (int | None): The label index to be ignored. Default: None
        topk (int | tuple[int], optional): If the predictions in ``topk``
            matches the target, the predictions will be regarded as
            correct ones. Defaults to 1.
        thresh (float, optional): If not None, predictions with scores under
            this threshold are considered incorrect. Default to None.

    Returns:
        float | tuple[float]: If the input ``topk`` is a single integer,
            the function will return a single float as accuracy. If
            ``topk`` is a tuple containing multiple integers, the
            function will return a tuple containing accuracies of
            each ``topk`` number.
    """
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = (topk,)
        return_single = True
    else:
        return_single = False

    maxk = max(topk)
    if pred.size(0) == 0:
        accu = [pred.new_tensor(0.) for i in range(len(topk))]
        return accu[0] if return_single else accu
    assert pred.ndim == target.ndim + 1
    assert pred.size(0) == target.size(0)
    assert maxk <= pred.size(1), \
        f'maxk {maxk} exceeds pred dimension {pred.size(1)}'
    pred_value, pred_label = pred.topk(maxk, dim=1)
    # transpose to shape (maxk, N, ...)
    pred_label = pred_label.transpose(0, 1)
    correct = pred_label.eq(target.unsqueeze(0).expand_as(pred_label))
    if thresh is not None:
        # Only prediction values larger than thresh are counted as correct
        correct = correct & (pred_value > thresh).t()

    target_mask = Union[Tuple[Tensor, ...], Tensor]
    if ignore_index is not None:
        target_mask = torch.nonzero(target != ignore_index, as_tuple=True)
        correct = correct[:, target_mask[0], target_mask[1], target_mask[2]]
    res = []
    eps = torch.finfo(torch.float32).eps
    for k in topk:
        # Avoid causing ZeroDivisionError when all pixels
        # of an image are ignored
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True) + eps
        if ignore_index is not None:
            total_num = target[target_mask].numel() + eps
        else:
            total_num = target.numel() + eps
        res.append(correct_k.mul_(100.0 / total_num))
    return res[0] if return_single else res


def loss_by_feat(self, seg_logits: Tensor,
                 batch_data_samples: SampleList) -> dict:
    """Compute segmentation loss.

    Args:
        seg_logits (Tensor): The output from decode head forward function.
        batch_data_samples (List[:obj:`SegDataSample`]): The seg
            data samples. It usually includes information such
            as `metainfo` and `gt_sem_seg`.

    Returns:
        dict[str, Tensor]: a dictionary of loss components
    """

    seg_label = self._stack_batch_gt(batch_data_samples)
    loss = dict()
    seg_logits = resize(
        input=seg_logits,
        size=seg_label.shape[2:],
        mode='bilinear',
        align_corners=self.align_corners)
    if self.sampler is not None:
        seg_weight = self.sampler.sample(seg_logits, seg_label)
    else:
        seg_weight = None
    seg_label = seg_label.squeeze(1)

    if not isinstance(self.loss_decode, nn.ModuleList):
        losses_decode = [self.loss_decode]
    else:
        losses_decode = self.loss_decode
    for loss_decode in losses_decode:
        if loss_decode.loss_name not in loss:
            loss[loss_decode.loss_name] = loss_decode(
                seg_logits,
                seg_label,
                weight=seg_weight,
                ignore_index=self.ignore_index)
        else:
            loss[loss_decode.loss_name] += loss_decode(
                seg_logits,
                seg_label,
                weight=seg_weight,
                ignore_index=self.ignore_index)

    loss['acc_seg'] = accuracy(
        seg_logits, seg_label, ignore_index=self.ignore_index)
    return loss


BaseDecodeHead.loss_by_feat = loss_by_feat
