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
import torch
from fairseq.data.audio.feature_transforms import (
    AudioFeatureTransform,
    register_audio_feature_transform,
)


@register_audio_feature_transform("delta_deltas")
class DeltaDeltas(AudioFeatureTransform):
    """Expand delta-deltas features from spectrum."""

    @classmethod
    def from_config_dict(cls, config=None):
        _config = {} if config is None else config
        return DeltaDeltas(_config.get("win_length", 5))

    def __init__(self, win_length=5):
        self.win_length = win_length

    def __repr__(self):
        return self.__class__.__name__

    def __call__(self, spectrogram):
        from torchaudio.functional import compute_deltas

        assert len(spectrogram.shape) == 2, "spectrogram must be a 2-D tensor."
        # spectrogram is T x F, while compute_deltas takes (…, F, T)
        spectrogram = torch.from_numpy(spectrogram).transpose(0, 1)
        delta = compute_deltas(spectrogram)
        delta_delta = compute_deltas(delta)

        out_feat = np.concatenate(
            [spectrogram, delta.numpy(), delta_delta.numpy()], axis=0
        )
        out_feat = np.transpose(out_feat)
        return out_feat
