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
# limitations under the License.

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# Modified by Linjie Deng
# --------------------------------------------------------
import torch

from utils.nms.cpu_nms import cpu_nms, cpu_soft_nms


def nms(dets, thresh, use_gpu=True):
    """Dispatch to either CPU or GPU NMS implementations."""
    if dets.shape[0] == 0:
        return []
    if dets.shape[1] == 5:
        raise NotImplementedError
    elif dets.shape[1] == 6:
        if torch.is_tensor(dets):
            dets = dets.cpu().detach().numpy()
        return cpu_nms(dets, thresh)
    else:
        raise NotImplementedError
