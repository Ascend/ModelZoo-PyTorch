# Copyright 2022 Huawei Technologies Co., Ltd
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
import torch_npu

class NpuClamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, self, min=None, max=None):
        ctx.save_for_backward(self)
        ctx.min = min
        ctx.max = max
        result = torch.clamp(self, min, max)
        return result

    @staticmethod
    def backward(ctx, grad):
        self = ctx.saved_tensors[0]
        min = ctx.min
        max = ctx.max
        if min is not None and max is not None:
            grad_input = grad * ((self >= min).type_as(grad) * (self <= max).type_as(grad))
        elif min is not None:
            grad_input = grad * (self >= min).type_as(grad)
        elif max is not None:
            grad_input = grad * (self <= max).type_as(grad)
        else:
            grad_input = grad
        return grad_input, None, None

def npu_clamp(self, min=None, max=None):
    return NpuClamp.apply(self, min, max)
