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
import torch


def set_index(ori_tensor, mask, new_value):
    if mask is None:
        return ori_tensor
    elif ori_tensor.size() == mask.size():
        return ori_tensor * (1 - mask) + new_value * mask
    elif len(ori_tensor.size()) == 3 and ori_tensor.size(0) == mask.size(0) and ori_tensor.size(2) == mask.size(1):
        mask = torch.unsqueeze(mask, 1)
        return ori_tensor * (1 - mask) + new_value * mask
    elif len(ori_tensor.size()) == 3 and ori_tensor.size(0) == mask.size(0) and ori_tensor.size(1) == mask.size(1):
        mask = torch.unsqueeze(mask, 2)
        return ori_tensor * (1 - mask) + new_value * mask
