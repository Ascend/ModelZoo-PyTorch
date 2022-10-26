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
import torch.nn.functional as F


def one_hot_encoding(gt, num_classes):
    """Change gt_label to one_hot encoding.

    If the shape has 2 or more
    dimensions, return it without encoding.
    Args:
        gt (Tensor): The gt label with shape (N,) or shape (N, */).
        num_classes (int): The number of classes.
    Return:
        Tensor: One hot gt label.
    """
    if gt.ndim == 1:
        # multi-class classification
        return F.one_hot(gt, num_classes=num_classes)
    else:
        # binary classification
        # example. [[0], [1], [1]]
        # multi-label classification
        # example. [[0, 1, 1], [1, 0, 0], [1, 1, 1]]
        return gt
