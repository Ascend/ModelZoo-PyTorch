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

# Copyright (c) Open-MMLab. All rights reserved.
from .flops_counter import get_model_complexity_info
from .fuse_conv_bn import fuse_conv_bn
from .weight_init import (INITIALIZERS, ConstantInit, KaimingInit, NormalInit,
                          PretrainedInit, UniformInit, XavierInit,
                          bias_init_with_prob, caffe2_xavier_init,
                          constant_init, initialize, kaiming_init, normal_init,
                          uniform_init, xavier_init)

__all__ = [
    'get_model_complexity_info', 'bias_init_with_prob', 'caffe2_xavier_init',
    'constant_init', 'kaiming_init', 'normal_init', 'uniform_init',
    'xavier_init', 'fuse_conv_bn', 'initialize', 'INITIALIZERS',
    'ConstantInit', 'XavierInit', 'NormalInit', 'UniformInit', 'KaimingInit',
    'PretrainedInit'
]
