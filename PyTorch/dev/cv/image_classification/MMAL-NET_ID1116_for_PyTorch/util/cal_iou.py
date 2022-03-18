#
# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================
#
import numpy as np
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
	NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
	torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')

def calculate_iou(coor1, coor2):
    """
    :param coor1:dtype = np.array, shape = [:,4]
    :param coor2:
    :return:
    """
    # 鎺掗櫎涓庨変腑鐨刟nchor box iou澶т簬闃堝肩殑anchor boxes
    start_max = np.maximum(coor1[:, 0:2], coor2[:, 0:2])  # [338,2]
    end_min = np.minimum(coor1[:, 2:4], coor2[:, 2:4])  # [338,2]
    lengths = end_min - start_max + 1  # [338,2]

    intersection = lengths[:, 0] * lengths[:, 1]
    intersection[np.logical_or(lengths[:, 0] < 0, lengths[:, 1] < 0)] = 0

    union = ((coor1[:, 2] - coor1[:, 0] + 1) * (coor1[:, 3] - coor1[:, 1] + 1)
             + (coor2[:, 2] - coor2[:, 0] + 1) * (coor2[:, 3] - coor2[:, 1] + 1)
             - intersection)

    iou = intersection / union  # (338,)

    return iou