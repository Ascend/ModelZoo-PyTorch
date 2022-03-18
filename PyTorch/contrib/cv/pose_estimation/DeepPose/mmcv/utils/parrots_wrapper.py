# Copyright 2020 Huawei Technologies Co., Ltd
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
from functools import partial

import torch

TORCH_VERSION = torch.__version__


def _get_cuda_home():
    if TORCH_VERSION == 'parrots':
        from parrots.utils.build_extension import CUDA_HOME
    else:
        from torch.utils.cpp_extension import CUDA_HOME
    return CUDA_HOME


def get_build_config():
    if TORCH_VERSION == 'parrots':
        from parrots.config import get_build_info
        return get_build_info()
    else:
        return torch.__config__.show()


def _get_conv():
    if TORCH_VERSION == 'parrots':
        from parrots.nn.modules.conv import _ConvNd, _ConvTransposeMixin
    else:
        from torch.nn.modules.conv import _ConvNd, _ConvTransposeMixin
    return _ConvNd, _ConvTransposeMixin


def _get_dataloader():
    if TORCH_VERSION == 'parrots':
        from torch.utils.data import DataLoader, PoolDataLoader
    else:
        from torch.utils.data import DataLoader
        PoolDataLoader = DataLoader
    return DataLoader, PoolDataLoader


def _get_extension():
    if TORCH_VERSION == 'parrots':
        from parrots.utils.build_extension import BuildExtension, Extension
        CppExtension = partial(Extension, cuda=False)
        CUDAExtension = partial(Extension, cuda=True)
    else:
        from torch.utils.cpp_extension import (BuildExtension, CppExtension,
                                               CUDAExtension)
    return BuildExtension, CppExtension, CUDAExtension


def _get_pool():
    if TORCH_VERSION == 'parrots':
        from parrots.nn.modules.pool import (_AdaptiveAvgPoolNd,
                                             _AdaptiveMaxPoolNd, _AvgPoolNd,
                                             _MaxPoolNd)
    else:
        from torch.nn.modules.pooling import (_AdaptiveAvgPoolNd,
                                              _AdaptiveMaxPoolNd, _AvgPoolNd,
                                              _MaxPoolNd)
    return _AdaptiveAvgPoolNd, _AdaptiveMaxPoolNd, _AvgPoolNd, _MaxPoolNd


def _get_norm():
    if TORCH_VERSION == 'parrots':
        from parrots.nn.modules.batchnorm import _BatchNorm, _InstanceNorm
        SyncBatchNorm_ = torch.nn.SyncBatchNorm2d
    else:
        from torch.nn.modules.instancenorm import _InstanceNorm
        from torch.nn.modules.batchnorm import _BatchNorm
        SyncBatchNorm_ = torch.nn.SyncBatchNorm
    return _BatchNorm, _InstanceNorm, SyncBatchNorm_


CUDA_HOME = _get_cuda_home()
_ConvNd, _ConvTransposeMixin = _get_conv()
DataLoader, PoolDataLoader = _get_dataloader()
BuildExtension, CppExtension, CUDAExtension = _get_extension()
_BatchNorm, _InstanceNorm, SyncBatchNorm_ = _get_norm()
_AdaptiveAvgPoolNd, _AdaptiveMaxPoolNd, _AvgPoolNd, _MaxPoolNd = _get_pool()


class SyncBatchNorm(SyncBatchNorm_):

    def _specify_ddp_gpu_num(self, gpu_size):
        if TORCH_VERSION != 'parrots':
            super()._specify_ddp_gpu_num(gpu_size)

    def _check_input_dim(self, input):
        if TORCH_VERSION == 'parrots':
            if input.dim() < 2:
                raise ValueError(
                    f'expected at least 2D input (got {input.dim()}D input)')
        else:
            super()._check_input_dim(input)
