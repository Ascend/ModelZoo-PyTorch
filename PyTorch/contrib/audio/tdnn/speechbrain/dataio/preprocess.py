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

"""Preprocessors for audio"""
import torch
import functools
from speechbrain.processing.speech_augmentation import Resample


class AudioNormalizer:
    """Normalizes audio into a standard format

    Arguments
    ---------
    sample_rate : int
        The sampling rate to which the incoming signals should be converted.
    mix : {"avg-to-mono", "keep"}
        "avg-to-mono" - add all channels together and normalize by number of
        channels. This also removes the channel dimension, resulting in [time]
        format tensor.
        "keep" - don't normalize channel information

    Example
    -------
    >>> import torchaudio
    >>> example_file = 'samples/audio_samples/example_multichannel.wav'
    >>> signal, sr = torchaudio.load(example_file, channels_first = False)
    >>> normalizer = AudioNormalizer(sample_rate=8000)
    >>> normalized = normalizer(signal, sr)
    >>> signal.shape
    torch.Size([33882, 2])
    >>> normalized.shape
    torch.Size([16941])

    NOTE
    ----
    This will also upsample audio. However, upsampling cannot produce meaningful
    information in the bandwidth which it adds. Generally models will not work
    well for upsampled data if they have not specifically been trained to do so.
    """

    def __init__(self, sample_rate=16000, mix="avg-to-mono"):
        self.sample_rate = sample_rate
        if mix not in ["avg-to-mono", "keep"]:
            raise ValueError(f"Unexpected mixing configuration {mix}")
        self.mix = mix
        self._cached_resample = functools.lru_cache(maxsize=12)(Resample)

    def __call__(self, audio, sample_rate):
        """Perform normalization

        Arguments
        ---------
        audio : tensor
            The input waveform torch tensor. Assuming [time, channels],
            or [time].
        """
        resampler = self._cached_resample(sample_rate, self.sample_rate)
        resampled = resampler(audio.unsqueeze(0)).squeeze(0)
        return self._mix(resampled)

    def _mix(self, audio):
        """Handle channel mixing"""
        flat_input = audio.dim() == 1
        if self.mix == "avg-to-mono":
            if flat_input:
                return audio
            return torch.mean(audio, 1)
        if self.mix == "keep":
            return audio
