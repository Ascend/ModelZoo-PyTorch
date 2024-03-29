# Copyright 2021 Huawei Technologies Co., Ltd
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
from .similar_api import max_unpool2d, max_unpool1d, MaxUnpool2d, MaxUnpool1d, SyncBatchNorm, \
    ApexDistributedDataParallel, Conv3d, get_device_properties, set_default_tensor_type, repeat_interleave, \
    TorchDistributedDataParallel, pad
__all__ = ["max_unpool1d", "max_unpool2d", "MaxUnpool1d", "MaxUnpool2d", "SyncBatchNorm", "ApexDistributedDataParallel",
           "Conv3d", "get_device_properties", "set_default_tensor_type", "repeat_interleave",
           "TorchDistributedDataParallel", "pad"]
