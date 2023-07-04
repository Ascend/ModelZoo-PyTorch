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


@register_audio_feature_transform("utterance_cmvn")
class UtteranceCMVN(AudioFeatureTransform):
    """Utterance-level CMVN (cepstral mean and variance normalization)"""

    @classmethod
    def from_config_dict(cls, config=None):
        _config = {} if config is None else config
        return UtteranceCMVN(
            _config.get("norm_means", True),
            _config.get("norm_vars", True),
        )

    def __init__(self, norm_means=True, norm_vars=True):
        self.norm_means, self.norm_vars = norm_means, norm_vars

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"(norm_means={self.norm_means}, norm_vars={self.norm_vars})"
        )

    def __call__(self, x):
        mean = x.mean(axis=0)
        square_sums = (x**2).sum(axis=0)

        if self.norm_means:
            x = np.subtract(x, mean)
        if self.norm_vars:
            var = square_sums / x.shape[0] - mean**2
            std = np.sqrt(np.maximum(var, 1e-10))
            x = np.divide(x, std)

        return x
