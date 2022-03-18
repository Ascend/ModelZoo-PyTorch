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

import numpy as np
import torch

from mmdet.utils.util_random import ensure_rng


def random_boxes(num=1, scale=1, rng=None):
    """Simple version of ``kwimage.Boxes.random``

    Returns:
        Tensor: shape (n, 4) in x1, y1, x2, y2 format.

    References:
        https://gitlab.kitware.com/computer-vision/kwimage/blob/master/kwimage/structs/boxes.py#L1390

    Example:
        >>> num = 3
        >>> scale = 512
        >>> rng = 0
        >>> boxes = random_boxes(num, scale, rng)
        >>> print(boxes)
        tensor([[280.9925, 278.9802, 308.6148, 366.1769],
                [216.9113, 330.6978, 224.0446, 456.5878],
                [405.3632, 196.3221, 493.3953, 270.7942]])
    """
    rng = ensure_rng(rng)

    tlbr = rng.rand(num, 4).astype(np.float32)

    tl_x = np.minimum(tlbr[:, 0], tlbr[:, 2])
    tl_y = np.minimum(tlbr[:, 1], tlbr[:, 3])
    br_x = np.maximum(tlbr[:, 0], tlbr[:, 2])
    br_y = np.maximum(tlbr[:, 1], tlbr[:, 3])

    tlbr[:, 0] = tl_x * scale
    tlbr[:, 1] = tl_y * scale
    tlbr[:, 2] = br_x * scale
    tlbr[:, 3] = br_y * scale

    boxes = torch.from_numpy(tlbr)
    return boxes
