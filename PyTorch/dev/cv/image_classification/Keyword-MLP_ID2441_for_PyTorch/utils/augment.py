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
import numba as nb
import librosa
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))


@nb.jit(nopython=True, cache=True)
def time_shift(wav: np.ndarray, sr: int, s_min: float, s_max: float) -> np.ndarray:
    """Time shift augmentation.
    Refer to https://www.kaggle.com/haqishen/augmentation-methods-for-audio#1.-Time-shifting.
    Changed np.r_ to np.hstack for numba support.

    Args:
        wav (np.ndarray): Waveform array of shape (n_samples,).
        sr (int): Sampling rate.
        s_min (float): Minimum fraction of a second by which to shift.
        s_max (float): Maximum fraction of a second by which to shift.
    
    Returns:
        wav_time_shift (np.ndarray): Time-shifted waveform array.
    """

    start = int(np.random.uniform(sr * s_min, sr * s_max))
    if start >= 0:
        wav_time_shift = np.hstack((wav[start:], np.random.uniform(-0.001, 0.001, start)))
    else:
        wav_time_shift = np.hstack((np.random.uniform(-0.001, 0.001, -start), wav[:start]))
    
    return wav_time_shift


def resample(x: np.ndarray, sr: int, r_min: float, r_max: float) -> np.ndarray:
    """Resamples waveform.

    Args:
        x (np.ndarray): Input waveform, array of shape (n_samples, ).
        sr (int): Sampling rate.
        r_min (float): Minimum percentage of resampling.
        r_max (float): Maximum percentage of resampling.
    """

    sr_new = sr * np.random.uniform(r_min, r_max)
    x = librosa.resample(x, sr, sr_new)
    return x, sr_new


@nb.jit(nopython=True, cache=True)
def spec_augment(mel_spec: np.ndarray, n_time_masks: int, time_mask_width: int, n_freq_masks: int, freq_mask_width: int):
    """Numpy implementation of spectral augmentation.

    Args:
        mel_spec (np.ndarray): Mel spectrogram, array of shape (n_mels, T).
        n_time_masks (int): Number of time bands.   
        time_mask_width (int): Max width of each time band.
        n_freq_masks (int): Number of frequency bands.
        freq_mask_width (int): Max width of each frequency band.

    Returns:
        mel_spec (np.ndarray): Spectrogram with random time bands and freq bands masked out.
    """
    
    offset, begin = 0, 0

    for _ in range(n_time_masks):
        offset = np.random.randint(0, time_mask_width)
        begin = np.random.randint(0, mel_spec.shape[1] - offset)
        mel_spec[:, begin: begin + offset] = 0.0
    
    for _ in range(n_freq_masks):
        offset = np.random.randint(0, freq_mask_width)
        begin = np.random.randint(0, mel_spec.shape[0] - offset)
        mel_spec[begin: begin + offset, :] = 0.0

    return mel_spec
