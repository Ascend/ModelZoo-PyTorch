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
from hashlib import sha256

import torch

import common.modes
import datasets._isr


def update_argparser(parser):
  datasets._isr.update_argparser(parser)
  parser.add_argument(
      '--noise_sigma',
      help='Noise sigma level for image denoising',
      default=25.0,
      type=float)
  parser.set_defaults(scale=1,)


class ImageDenoisingDataset(datasets._isr.ImageSuperResolutionDataset):

  def __init__(self, mode, params, image_files):
    super(ImageDenoisingDataset, self).__init__(mode, params, image_files,
                                                image_files)

  def __getitem__(self, index):
    x, y = super(ImageDenoisingDataset, self).__getitem__(index)
    if self.mode == common.modes.TRAIN:
      noise = torch.randn_like(x) * (self.params.noise_sigma / 255.0)
    else:
      image_file = self.lr_files[index][0]
      seed = np.frombuffer(
          sha256(image_file.encode('utf-8')).digest(), dtype='uint32')
      rstate = np.random.RandomState(seed)
      noise = rstate.normal(0, self.params.noise_sigma / 255.0, x.shape)
      noise = torch.from_numpy(noise).float()
    x += noise
    return x, y


class ImageDenoisingHdf5Dataset(datasets._isr.ImageSuperResolutionHdf5Dataset):

  def __init__(
      self,
      mode,
      params,
      image_files,
      image_cache_file,
      lib_hdf5='h5py',
  ):
    super(ImageDenoisingHdf5Dataset, self).__init__(
        mode,
        params,
        image_files,
        image_files,
        image_cache_file,
        image_cache_file,
        lib_hdf5=lib_hdf5,
    )

  def __getitem__(self, index):
    x, y = super(ImageDenoisingHdf5Dataset, self).__getitem__(index)
    if self.mode == common.modes.TRAIN:
      noise = torch.randn_like(x) * (self.params.noise_sigma / 255.0)
    else:
      image_file = self.lr_files[index][0]
      seed = np.frombuffer(
          sha256(image_file.encode('utf-8')).digest(), dtype='uint32')
      rstate = np.random.RandomState(seed)
      noise = rstate.normal(0, self.params.noise_sigma / 255.0, x.shape)
      noise = torch.from_numpy(noise).float()
    return x + noise, y
