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

"""
    @author fangyi.zhang@vipl.ict.ac.cn
"""
import numpy as np

def determine_thresholds(confidence, resolution=100):
    """choose threshold according to confidence

    Args:
        confidence: list or numpy array or numpy array
        reolution: number of threshold to choose

    Restures:
        threshold: numpy array
    """
    if isinstance(confidence, list):
        confidence = np.array(confidence)
    confidence = confidence.flatten()
    confidence = confidence[~np.isnan(confidence)]
    confidence.sort()

    assert len(confidence) > resolution and resolution > 2

    thresholds = np.ones((resolution))
    thresholds[0] = - np.inf
    thresholds[-1] = np.inf
    delta = np.floor(len(confidence) / (resolution - 2))
    idxs = np.linspace(delta, len(confidence)-delta, resolution-2, dtype=np.int32)
    thresholds[1:-1] =  confidence[idxs]
    return thresholds
