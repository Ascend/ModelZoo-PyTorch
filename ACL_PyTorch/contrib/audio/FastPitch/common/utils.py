# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import shutil
import warnings
from pathlib import Path
from typing import Optional

import librosa
import numpy as np

import torch
from scipy.io.wavfile import read


def mask_from_lens(lens, max_len: Optional[int] = None):
    if max_len is None:
        max_len = lens.max()
    ids = torch.arange(0, max_len, device=lens.device, dtype=lens.dtype)
    mask = torch.lt(ids, lens.unsqueeze(1))
    return mask


def load_wav_to_torch(full_path, force_sampling_rate=None):
    if force_sampling_rate is not None:
        data, sampling_rate = librosa.load(full_path, sr=force_sampling_rate)
    else:
        sampling_rate, data = read(full_path)

    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(dataset_path, fnames, has_speakers=False, split="|"):
    def split_line(root, line):
        parts = line.strip().split(split)
        if has_speakers:
            paths, non_paths = parts[:-2], parts[-2:]
        else:
            paths, non_paths = parts[:-1], parts[-1:]
        return tuple(str(Path(root, p)) for p in paths) + tuple(non_paths)

    fpaths_and_text = []
    for fname in fnames:
        with open(fname, encoding='utf-8') as f:
            fpaths_and_text += [split_line(dataset_path, line) for line in f]
    return fpaths_and_text


def stats_filename(dataset_path, filelist_path, feature_name):
    stem = Path(filelist_path).stem
    return Path(dataset_path, f'{feature_name}_stats__{stem}.json')


def to_gpu(x):
    x = x.contiguous()
    if torch.cuda.is_available():
       x = x.cuda(non_blocking=True)
    # x = x.to("npu")
    return torch.autograd.Variable(x)


def to_device_async(tensor, device):
    return tensor.to(device, non_blocking=True)


def to_numpy(x):
    return x.cpu().numpy() if isinstance(x, torch.Tensor) else x


def prepare_tmp(path):
    if path is None:
        return
    p = Path(path)
    if p.is_dir():
        warnings.warn(f'{p} exists. Removing...')
        shutil.rmtree(p, ignore_errors=True)
    p.mkdir(parents=False, exist_ok=False)
