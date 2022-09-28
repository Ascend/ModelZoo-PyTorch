# Copyright 2022 Huawei Technologies Co., Ltd.
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
# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.wrappers import NewEmptyTensorOp, obsolete_torch_version

if torch.__version__ == 'parrots':
    TORCH_VERSION = torch.__version__
else:
    # torch.__version__ could be 1.3.1+cu92, we only need the first two
    # for comparison
    TORCH_VERSION = tuple(int(x) for x in torch.__version__.split('.')[:2])


def adaptive_avg_pool2d(input, output_size):
    """Handle empty batch dimension to adaptive_avg_pool2d.

    Args:
        input (tensor): 4D tensor.
        output_size (int, tuple[int,int]): the target output size.
    """
    if input.numel() == 0 and obsolete_torch_version(TORCH_VERSION, (1, 9)):
        if isinstance(output_size, int):
            output_size = [output_size, output_size]
        output_size = [*input.shape[:2], *output_size]
        empty = NewEmptyTensorOp.apply(input, output_size)
        return empty
    else:
        return F.adaptive_avg_pool2d(input, output_size)


class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    """Handle empty batch dimension to AdaptiveAvgPool2d."""

    def forward(self, x):
        # PyTorch 1.9 does not support empty tensor inference yet
        if x.numel() == 0 and obsolete_torch_version(TORCH_VERSION, (1, 9)):
            output_size = self.output_size
            if isinstance(output_size, int):
                output_size = [output_size, output_size]
            else:
                output_size = [
                    v if v is not None else d
                    for v, d in zip(output_size,
                                    x.size()[-2:])
                ]
            output_size = [*x.shape[:2], *output_size]
            empty = NewEmptyTensorOp.apply(x, output_size)
            return empty

        return super().forward(x)
