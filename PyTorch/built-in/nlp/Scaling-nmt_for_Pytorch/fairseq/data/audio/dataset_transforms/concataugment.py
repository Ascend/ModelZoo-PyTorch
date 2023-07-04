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


from typing import List
import numpy as np

from fairseq.data.audio.dataset_transforms import (
    AudioDatasetTransform,
    register_audio_dataset_transform,
)

_DEFAULTS = {"rate": 0.25, "max_tokens": 3000, "attempts": 5}


@register_audio_dataset_transform("concataugment")
class ConcatAugment(AudioDatasetTransform):
    @classmethod
    def from_config_dict(cls, config=None):
        _config = {} if config is None else config
        return ConcatAugment(
            _config.get("rate", _DEFAULTS["rate"]),
            _config.get("max_tokens", _DEFAULTS["max_tokens"]),
            _config.get("attempts", _DEFAULTS["attempts"]),
        )

    def __init__(
        self,
        rate=_DEFAULTS["rate"],
        max_tokens=_DEFAULTS["max_tokens"],
        attempts=_DEFAULTS["attempts"],
    ):
        self.rate, self.max_tokens, self.attempts = rate, max_tokens, attempts

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + ", ".join(
                [
                    f"rate={self.rate}",
                    f"max_tokens={self.max_tokens}",
                    f"attempts={self.attempts}",
                ]
            )
            + ")"
        )

    def find_indices(self, index: int, n_frames: List[int], n_samples: int):
        # skip conditions: application rate, max_tokens limit exceeded
        if np.random.random() > self.rate:
            return [index]
        if self.max_tokens and n_frames[index] > self.max_tokens:
            return [index]

        # pick second sample to concatenate
        for _ in range(self.attempts):
            index2 = np.random.randint(0, n_samples)
            if index2 != index and (
                not self.max_tokens
                or n_frames[index] + n_frames[index2] < self.max_tokens
            ):
                return [index, index2]

        return [index]
