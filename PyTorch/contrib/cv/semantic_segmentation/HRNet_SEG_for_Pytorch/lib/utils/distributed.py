# ------------------------------------------------------------------------------
# Copyright 2021 Huawei Technologies Co., Ltd
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
# limitations under the License.)
# ------------------------------------------------------------------------------

import torch
import torch.distributed as torch_dist

def is_distributed():
    return torch_dist.is_initialized()

def get_world_size():
    if not torch_dist.is_initialized():
        return 1
    return torch_dist.get_world_size()

def get_rank():
    if not torch_dist.is_initialized():
        return 0
    return torch_dist.get_rank()