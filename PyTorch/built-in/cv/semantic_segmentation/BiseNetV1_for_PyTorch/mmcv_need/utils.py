# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright 2020 Huawei Technologies Co., Ltd
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
# --------------------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.utils import IS_CUDA_AVAILABLE, IS_MLU_AVAILABLE, IS_MPS_AVAILABLE, IS_NPU_AVAILABLE


def get_device() -> str:
    """Returns the currently existing device type.

    Returns:
        str: npu | cuda | mlu | mps | cpu.
    """
    if IS_NPU_AVAILABLE:
        return 'npu'
    elif IS_CUDA_AVAILABLE:
        return 'cuda'
    elif IS_MLU_AVAILABLE:
        return 'mlu'
    elif IS_MPS_AVAILABLE:
        return 'mps'
    else:
        return 'cpu'
