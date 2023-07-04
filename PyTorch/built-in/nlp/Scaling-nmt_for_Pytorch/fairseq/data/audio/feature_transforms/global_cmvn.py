# coding:utf-8
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


import numpy as np
from fairseq.data.audio.feature_transforms import (
    AudioFeatureTransform,
    register_audio_feature_transform,
)


@register_audio_feature_transform("global_cmvn")
class GlobalCMVN(AudioFeatureTransform):
    """Global CMVN (cepstral mean and variance normalization). The global mean
    and variance need to be pre-computed and stored in NumPy format (.npz)."""

    @classmethod
    def from_config_dict(cls, config=None):
        _config = {} if config is None else config
        return GlobalCMVN(_config.get("stats_npz_path"))

    def __init__(self, stats_npz_path):
        self.stats_npz_path = stats_npz_path
        stats = np.load(stats_npz_path)
        self.mean, self.std = stats["mean"], stats["std"]

    def __repr__(self):
        return self.__class__.__name__ + f'(stats_npz_path="{self.stats_npz_path}")'

    def __call__(self, x):
        x = np.subtract(x, self.mean)
        x = np.divide(x, self.std)
        return x
