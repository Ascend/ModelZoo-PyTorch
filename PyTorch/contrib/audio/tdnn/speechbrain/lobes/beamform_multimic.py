#     Copyright 2021 Huawei Technologies Co., Ltd
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
#

"""Beamformer for multi-mic processing.

Authors
 * Nauman Dawalatabad
"""
import torch
from speechbrain.processing.features import (
    STFT,
    ISTFT,
)

from speechbrain.processing.multi_mic import (
    Covariance,
    GccPhat,
    DelaySum,
)


class DelaySum_Beamformer(torch.nn.Module):
    """Generate beamformed signal from multi-mic data using DelaySum beamforming.

    Arguments
    ---------
    sampling_rate : int (default: 16000)
        Sampling rate of audio signals.
    """

    def __init__(self, sampling_rate=16000):
        super().__init__()
        self.fs = sampling_rate
        self.stft = STFT(sample_rate=self.fs)
        self.cov = Covariance()
        self.gccphat = GccPhat()
        self.delaysum = DelaySum()
        self.istft = ISTFT(sample_rate=self.fs)

    def forward(self, mics_signals):
        """Returns beamformed signal using multi-mic data.

        Arguments
        ---------
        mics_sginal : tensor
            Set of audio signals to be transformed.
        """
        with torch.no_grad():

            Xs = self.stft(mics_signals)
            XXs = self.cov(Xs)
            tdoas = self.gccphat(XXs)
            Ys_ds = self.delaysum(Xs, tdoas)
            sig = self.istft(Ys_ds)

        return sig
