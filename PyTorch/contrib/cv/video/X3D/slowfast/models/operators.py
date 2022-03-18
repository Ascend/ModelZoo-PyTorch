# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


"""Custom operators."""

import torch
import torch.nn as nn


class Swish(nn.Module):
    """Swish activation function: x * sigmoid(x)."""

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return SwishEfficient.apply(x)


class SwishEfficient(torch.autograd.Function):
    """Swish activation function: x * sigmoid(x)."""

    @staticmethod
    def forward(ctx, x):
        result = x * torch.sigmoid(x)
        ctx.save_for_backward(x)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        sigmoid_x = torch.sigmoid(x)
        return grad_output * (sigmoid_x * (1 + x * (1 - sigmoid_x)))


class SE(nn.Module):
    """Squeeze-and-Excitation (SE) block w/ Swish: AvgPool, FC, Swish, FC, Sigmoid."""

    def _round_width(self, width, multiplier, min_width=8, divisor=8):
        """
        Round width of filters based on width multiplier
        Args:
            width (int): the channel dimensions of the input.
            multiplier (float): the multiplication factor.
            min_width (int): the minimum width after multiplication.
            divisor (int): the new width should be dividable by divisor.
        """
        if not multiplier:
            return width

        width *= multiplier
        min_width = min_width or divisor
        width_out = max(
            min_width, int(width + divisor / 2) // divisor * divisor
        )
        if width_out < 0.9 * width:
            width_out += divisor
        return int(width_out)

    def __init__(self, dim_in, ratio, relu_act=True):
        """
        Args:
            dim_in (int): the channel dimensions of the input.
            ratio (float): the channel reduction ratio for squeeze.
            relu_act (bool): whether to use ReLU activation instead
                of Swish (default).
            divisor (int): the new width should be dividable by divisor.
        """
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        dim_fc = self._round_width(dim_in, ratio)
        self.fc1 = nn.Conv3d(dim_in, dim_fc, 1, bias=True)
        self.fc1_act = nn.ReLU() if relu_act else Swish()
        self.fc2 = nn.Conv3d(dim_fc, dim_in, 1, bias=True)

        self.fc2_sig = nn.Sigmoid()

    def forward(self, x):
        # x_in = x
        
        # for module in self.children():
            # x = module(x)
        # return x_in * x
        
        
        # x shape (40,54,13,40,40)
        
        
        
        x_in = x        
        x = self.avg_pool(x)  #  (40,54,1,1,1)  right
        # print("SE conv3d input shape", x.shape)
        
       
        
        # print("error input shape",x.shape)
        x = self.fc1(x)    # (40,8,1,1,1)  error
        # print("weight shape",self.fc1.weight.shape)
        x = self.fc1_act(x)  # (40,8,1,1,1)
        
       
        
        # x = torch.rand(16,8,1,1,1).npu()
        # x.requires_grad_()
        
        # print("error input shape",x.shape)
        x = self.fc2(x)  # (40,54,1,1,1)   error
        # print("weight shape",self.fc2.weight.shape)
        
        
        # print("\n")
        
        x = self.fc2_sig(x)  # (40,54,1,1,1)
        
        return x_in * x
