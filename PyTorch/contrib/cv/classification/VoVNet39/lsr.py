# Copyright (c) 2020, Huawei Technologies.All rights reserved.
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
if torch.__version__ >= '1.8':
    import torch_npu
import torch.nn as nn


class NPULabelSmoothingCrossEntropy(nn.Module):
    """CrossEntropy with LabelSmoothing using npu api.

    Paper: [Rethinking the Inception Architecture for Computer Vision]
    https://arxiv.org/pdf/1512.00567.pdf

    Args:
        smooth_factor (float): default 0. If label_smoothing using, using 0.1([0, 1]) instead.
        num_classes (float): classes numbers using for onehot.

    Returns:
        float: tensors of shape (k, 5) and (k, 1). Labels are 0-based.
    """

    def __init__(self, num_classes=1000, smooth_factor=0.):
        super(NPULabelSmoothingCrossEntropy, self).__init__()
        self.on_value = 1.0 - smooth_factor
        self.off_value = 1.0 * smooth_factor / (num_classes - 1)

    def forward(self, pred, target):
        one_hot_label = torch_npu.npu_one_hot(target.int(), -1, pred.size(1), self.on_value, self.off_value)
        loss = torch_npu.npu_softmax_cross_entropy_with_logits(pred, one_hot_label)

        loss = torch.mean(loss, [0], keepdim=False, dtype=torch.float32)
        return loss


class GPULabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, num_classes=1000, smooth_factor=0.0):
        """
        Constructor for the LabelSmoothing module.

        :param smoothing: label smoothing factor
        """
        super(GPULabelSmoothingCrossEntropy, self).__init__()
        self.confidence = 1.0 - smooth_factor
        self.smoothing = smooth_factor

    def forward(self, x, target):
        target = target.to(torch.int64)

        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1).to(torch.int64))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def get_label_smoothing_cross_entropy(device, num_classes=1000, smooth_factor=0.0):
    if device.split(':')[0] == 'npu':
        return NPULabelSmoothingCrossEntropy(num_classes, smooth_factor)
    else:
        return GPULabelSmoothingCrossEntropy(num_classes, smooth_factor)


if __name__ == '__main__':
    x = torch.randn(2, 10)
    x.requires_grad = True
    y = torch.randint(0, 10, size=(2,))

    torch.npu.set_device(0)
    x = x.npu()
    y = y.npu()
    m = NPULabelSmoothingCrossEntropy(10)
    l = m(x, y)
    l.backward()
    print('test ce ok, loss is ', l)

    m = NPULabelSmoothingCrossEntropy(10, 0.1)
    l = m(x, y)
    l.backward()
    print('test lsce ok, loss is ', l)

