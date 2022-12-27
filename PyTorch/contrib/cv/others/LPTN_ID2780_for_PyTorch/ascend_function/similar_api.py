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
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
import itertools
import os
import re

import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _triple


class StubDeviceProperties(object):
    """
    Stub class for torch.npu.get_device_properties
    If the device information is used, the user needs to modify this class.
    """
    def __init__(self):
        self.name = ""
        self.major = 8
        self.minor = 1
        self.total_memory = 15171
        self.multi_processor_count = 10


def repeat_interleave(self, repeats, dim=None):
    """
    Alternative implementation of torch.repeat_interleave to ensure consistent output,
    but the efficiency may decrease.
    Args:
    The description of arguments refers to torch.repeat_interleave

    Returns:
    The description of return value refers to torch.repeat_interleave
    """
    if not isinstance(repeats, (int, torch.Tensor)):
        raise RuntimeError("repeats must be int or Tensor")
    if dim is not None:
        return _repeat_interleave_with_dim(self, repeats, dim)
    self_reshape = self.reshape(-1, 1)
    if isinstance(repeats, int):
        _repeats = [repeats] * self_reshape.shape[0]
    else:
        if repeats.dim() != 0 and repeats.dim() != 1:
            raise RuntimeError("repeats must be 0-dim or 1-dim tensor")
        _repeats = repeats.cpu().numpy().tolist()
    return _repeat_interleave(self_reshape, _repeats)


def _repeat_interleave(self, repeats):
    if len(repeats) != self.shape[0]:
        raise RuntimeError("RuntimeError: repeats must have the same size as input along dim")
    new_tensor = torch.zeros(0, dtype=self.dtype, device=self.device)
    for i in range(self.shape[0]):
        sub_new_tensor = torch.cat(([self[i]] * repeats[i]))
        new_tensor = torch.cat((new_tensor, sub_new_tensor))
    return new_tensor.reshape(-1)


def _repeat_interleave_with_dim(self, repeats, dim=0):
    if dim >= self.dim() or dim < -1 * self.dim():
        raise IndexError(f"Dimension out of range (expected to be in range of "
                         f"[{-1 * self.dim()}, {self.dim()}], but got {dim})")
    _dim = dim if dim >= 0 else (dim + self.dim())
    if isinstance(repeats, int):
        _repeats = [repeats] * self.shape[_dim]
    else:
        if repeats.dim() != 0 and repeats.dim() != 1:
            raise RuntimeError("repeats must be 0-dim or 1-dim tensor")
        _repeats = repeats.cpu().numpy().tolist()
    if len(_repeats) != self.shape[_dim]:
        raise RuntimeError("RuntimeError: repeats must have the same size as input along dim")
    _sum = sum(_repeats)
    new_shape = list(self.shape)
    new_shape[_dim] = _sum
    new_tensor = torch.zeros(*new_shape, dtype=self.dtype, device=self.device)
    if dim == 0:
        list_repeats = list(_repeats)
        _repeat_tensor(list_repeats, new_tensor, self)
    else:
        list_range = (range(n) for n in new_shape[0:_dim])
        for tensor_idx_tuple in itertools.product(*list_range):
            sub_new_tensor = new_tensor[tuple(tensor_idx_tuple)]
            sub_tensor = self[tuple(tensor_idx_tuple)]
            list_repeats = list(_repeats)
            _repeat_tensor(list_repeats, sub_new_tensor, sub_tensor)
    return new_tensor


def _repeat_tensor(list_repeats, new_tensor, input_tensor):
    tensor_idx = 0
    for repeat_idx, repeat_num in enumerate(list_repeats):
        for _ in range(repeat_num):
            new_tensor[tensor_idx] = new_tensor[tensor_idx] + input_tensor[repeat_idx]
            tensor_idx += 1


def pad(input, pad, mode='constant', value=0):
    """
    Use to replace torch.nn.functional.pad.
    Function pad on npu now only support constant mode and 4-dim or 5-dim tensor.
    Args:
    Refers to torch.nn.functional.pad.

    Returns:
    pad result on constant mode
    """
    sub_dim = 4 - input.dim()
    if sub_dim <= 0:
        return F.pad(input, pad, 'constant', value)
    sub_shape = [1] * sub_dim
    new_input = input.reshape(*sub_shape, *input.shape)
    output = F.pad(new_input, pad, 'constant', value)
    return output.reshape(*output.shape[sub_dim:])


def set_default_tensor_type(type):
    """
    NPU shares dtypes with CPU. So, param which like NPU dtypes will be replaced with CPU dtypes here.But this won't
    affect the operation of tensor on NPU.
    """
    type_str = str(type)
    match_pattern = re.compile("'(.*)'")
    match_res = match_pattern.findall(type_str)
    if match_res:
        type_str = match_res[0]
    type_str = type_str.replace('torch.npu', 'torch')
    torch.set_default_tensor_type(type_str)


def get_device_properties(device):
    """
    Use to replace torch.cuda.get_device_properties.
    torch.npu.get_device_properties is not support for the time being.

    Returns:
    Class StubDeviceProperties
    """
    return StubDeviceProperties()


class Conv3d(torch.nn.Conv3d):
    """
    Use to replace torch.nn.Conv3d.
    This class will modify delation to 1,
    and expand the size of the convolution kernel to ensure that the output shape is consistent.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        kernel_size = _triple(kernel_size)
        stride_triple = _triple(stride)
        padding_triple = _triple(padding)
        dilation_triple = _triple(dilation)
        kernel_size = (kernel_size[0] + (dilation_triple[0] - 1) * (kernel_size[0] - 1),
                       kernel_size[1] + (dilation_triple[1] - 1) * (kernel_size[1] - 1),
                       kernel_size[2] + (dilation_triple[2] - 1) * (kernel_size[2] - 1))
        dilation = (1, 1, 1)
        super(Conv3d, self).__init__(
            in_channels, out_channels, kernel_size, stride_triple,
            padding_triple, dilation, groups, bias, padding_mode)


class SyncBatchNorm(torch.nn.SyncBatchNorm):
    """
    Use to replace torch.nn.SyncBatchNorm.
    The forward method of this class is consistent with the process where need_sync is false in torch.nn.SyncBatchNorm.
    """

    @classmethod
    def convert_sync_batchnorm(cls, module, process_group=None):
        """
        Use to replace torch.nn.SyncBatchNorm.convert_sync_batchnorm.
        Args:
        The description of arguments refers to torch.nn.SyncBatchNorm.convert_sync_batchnorm.

        Returns:
        return module without doing any operation.
        """
        return module

    def forward(self, input):
        self._check_input_dim(input)
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)


class ApexDistributedDataParallel(torch.nn.parallel.DistributedDataParallel):
    """
    Use to replace apex.parallel.DistributedDataParallel.
    Convert to torch.nn.parallel.DistributedDataParallel, with environment variables LOCAL_RANK as device_ids.
    """
    def __init__(self, module, message_size=10000000, delay_allreduce=False,
                 shared_param=None, allreduce_trigger_params=None, retain_allreduce_buffers=False,
                 allreduce_always_fp32=False, num_allreduce_streams=1, allreduce_communicators=None,
                 gradient_average=True, gradient_predivide_factor=1.0, gradient_average_split_factor=None,
                 prof=False):
        device_ids = os.environ['LOCAL_RANK'] if 'LOCAL_RANK' in os.environ else 0
        super(ApexDistributedDataParallel, self).__init__(module=module, device_ids=[device_ids],
                                                          broadcast_buffers=False)


class TorchDistributedDataParallel(torch.nn.parallel.DistributedDataParallel):
    """
    Use to replace torch.nn.parallel.DistributedDataParallel.
    Convert argument broadcast_buffers to False.
    """
    def __init__(self, module, device_ids=None,
                 output_device=None, dim=0, broadcast_buffers=True,
                 process_group=None, bucket_cap_mb=25,
                 find_unused_parameters=False,
                 check_reduction=False):
        super(TorchDistributedDataParallel, self).__init__(module=module, device_ids=device_ids,
                                                           output_device=output_device,
                                                           dim=dim, broadcast_buffers=False,
                                                           process_group=process_group,
                                                           bucket_cap_mb=bucket_cap_mb,
                                                           find_unused_parameters=find_unused_parameters,
                                                           check_reduction=check_reduction)
