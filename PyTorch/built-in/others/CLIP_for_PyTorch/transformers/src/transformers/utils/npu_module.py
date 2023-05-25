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

from collections.abc import Mapping
from typing import Any, Union

import torch
import torch.nn as nn
import torch_npu
from math import inf

class MatmulApply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, self, mat2):
        # y: a * b^T
        ctx.save_for_backward(self, mat2)
        result = torch.matmul(self, mat2.transpose(-2, -1))
        return result.detach()

    @staticmethod
    def backward(ctx, grad):
        # da: grad * b
        # db: grad^T * a
        self, mat2 = ctx.saved_tensors
        self_grad = torch_npu.npu_bmmV2(grad, mat2, [])
        mat2_grad = torch_npu.npu_bmmV2(grad.transpose(-2, -1), self, [])
        return self_grad, mat2_grad


def Matmul_transpose(tensor1, tensor2):
    return MatmulApply.apply(tensor1, tensor2)


class NpuLinear(nn.Linear):
    def forward(self, input):
        return torch_npu.npu_linear(input, self.weight, self.bias)

def clip_grad_norm_fused(combined_grads, combined_grad_masks, max_norm, norm_type):
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    tmp_lst = []
    if norm_type == inf:
        for combined_grad, combined_grad_mask in zip(combined_grads, combined_grad_masks):
            if combined_grad is not None:
                tmp_lst.append(combined_grad.float().abs().mul_(combined_grad_mask).max())
        total_norm = max(tmp_lst)
    else:
        for combined_grad, combined_grad_mask in zip(combined_grads, combined_grad_masks):
            if combined_grad is not None:
                combined_grad = combined_grad.float()
                tmp_lst.append(combined_grad.mul(combined_grad).mul_(combined_grad_mask).sum())
        total_norm = torch.stack(tmp_lst).sum().pow(1/norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for combined_grad in combined_grads:
            if combined_grad is not None:
                combined_grad.mul_(clip_coef)
    return total_norm

def clip_optimizer_grad_norm_fused(self, max_norm, norm_type=2):
    combined_grads = self.get_optimizer_combined_grads()
    combined_grad_masks = self.get_optimizer_combined_grad_masks()
    total_norm = clip_grad_norm_fused(combined_grads, combined_grad_masks, max_norm, norm_type)
    return total_norm

class Data_prefetcher():
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.npu.Stream()
        self.preload()

    def prepare_input(self, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self.prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self.prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = dict(device=self.device, non_blocking=True)
            return data.to(**kwargs)
        return data

    def preload(self):
        try:
            self.inputs = next(self.loader)
        except StopIteration:
            self.inputs = None
            return

        with torch.npu.stream(self.stream):
            self.inputs = self.prepare_input(self.inputs)
            if len(self.inputs) == 0:
                raise ValueError(
                    "The batch received was empty, your model won't be able to train on it. Double-check that your "
                    f"training dataset contains keys expected by the model: {','.join(self._signature_columns)}."
                )

    def next(self):
        torch.npu.current_stream().wait_stream(self.stream)
        inputs = self.inputs
        self.preload()
        return inputs

    
