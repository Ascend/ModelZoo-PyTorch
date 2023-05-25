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
if torch.__version__ >= '1.8':
    import torch_npu
import torch.nn as nn


def accuracy(pred, target, topk=1, thresh=None):
    """Calculate accuracy according to the prediction and target.

    Args:
        pred (torch.Tensor): The model prediction, shape (N, num_class, ...)
        target (torch.Tensor): The target of each prediction, shape (N, , ...)
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
        topk = (topk, )
        return_single = True
    else:
        return_single = False

    maxk = max(topk)
    # weik add start
    # when topk==1, topk is replaced by argmax
    assert maxk == 1
    # TODO: topk is set with other value
    # weik add end

    if pred.size(0) == 0:
        accu = [pred.new_tensor(0.) for i in range(len(topk))]
        return accu[0] if return_single else accu
    assert pred.ndim == target.ndim + 1
    assert pred.size(0) == target.size(0)
    assert maxk <= pred.size(1), \
        f'maxk {maxk} exceeds pred dimension {pred.size(1)}'

    # print("[bei's tag] pred.topk(maxk, dim=1)", pred.shape, pred.dtype, pred.storage().npu_format(), maxk)
    # pred_value, pred_label = pred.topk(maxk, dim=1)
    # weik add start
    pred_label = torch_npu.npu_format_cast(pred, 2).argmax(dim=1, keepdim=True)
    # print(pred_label.shape, pred_label.dtype, pred_label.storage().npu_format())
    torch.npu.synchronize()
    # pred_label = pred_label.npu_format_cast(2)
    # TODO: get pred_val when thresh is used
    # print("[bei's tag] torch.zeros_like(pred)", pred.shape, pred.dtype, pred.storage().npu_format())
    pred_value = torch.zeros_like(pred)
    torch.npu.synchronize()
    # print("[bei's tag] torch.zeros_like(pred) done")
    # weik add end

    # transpose to shape (maxk, N, ...)
    pred_label = pred_label.transpose(0, 1)
    
    # print("[bei's tag] pred_label", pred_label.shape, pred_label.dtype, pred_label.storage().npu_format())
    # print("[bei's tag] target", target.shape, target.dtype, target.storage().npu_format())
    correct = pred_label.eq(target.unsqueeze(0).expand_as(pred_label))
    if thresh is not None:
        # Only prediction values larger than thresh are counted as correct
        correct = correct & (pred_value > thresh).t()
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / target.numel()))
    return res[0] if return_single else res


class Accuracy(nn.Module):
    """Accuracy calculation module."""

    def __init__(self, topk=(1, ), thresh=None):
        """Module to calculate the accuracy.

        Args:
            topk (tuple, optional): The criterion used to calculate the
                accuracy. Defaults to (1,).
            thresh (float, optional): If not None, predictions with scores
                under this threshold are considered incorrect. Default to None.
        """
        super().__init__()
        self.topk = topk
        self.thresh = thresh

    def forward(self, pred, target):
        """Forward function to calculate accuracy.

        Args:
            pred (torch.Tensor): Prediction of models.
            target (torch.Tensor): Target for each prediction.

        Returns:
            tuple[float]: The accuracies under different topk criterions.
        """
        return accuracy(pred, target, self.topk, self.thresh)
