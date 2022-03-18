# Copyright (c) 2020, Huawei Technologies.All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from change_data_ptr import change_data_ptr

def combine_npu(list_of_tensor, require_copy_value = True):
    total_numel = 0
    for tensor in list_of_tensor:
        total_numel += tensor.storage().size()

    if total_numel == 0:
        return None
    
    dtype = list_of_tensor[0].dtype
    combined_tensor = torch.zeros(total_numel, dtype=dtype).npu()

    idx = 0
    if require_copy_value:
        for tensor in list_of_tensor:
            temp = tensor.clone()
            change_data_ptr(tensor, combined_tensor, idx)
            tensor.copy_(temp)
            idx += temp.storage().size()
    else:
        for tensor in list_of_tensor:
            change_data_ptr(tensor, combined_tensor, idx)
            idx += tensor.storage().size()
    return combined_tensor
