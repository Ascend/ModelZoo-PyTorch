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
"""Image Super-resolution dataset."""

import os
import random
import numpy as np
from PIL import Image
from PIL import ImageOps
from skimage.io import imread
from skimage import img_as_float, img_as_ubyte
import logging
import torch.utils.data as data
import torchvision.transforms as transforms

import common.modes
import common.io
from common.images import imresize
import datasets


def update_argparser(parser):
  datasets.update_argparser(parser)
  parser.add_argument(
      '--scale',
      help='Scale factor for image super-resolution.',
      default=2,
      type=int)
  parser.add_argument(
      '--lr_patch_size',
      help='Number of pixels in height or width of LR patches.',
      default=48,
      type=int)
  parser.add_argument(
      '--ignored_boundary_size',
      help='Number of ignored boundary pixels of LR patches.',
      default=2,
      type=int)
  parser.add_argument(
      '--num_patches',
      help='Number of sampling patches per image for training.',
      default=100,
      type=int)
  parser.set_defaults(
      train_batch_size=16,
      eval_batch_size=1,
      image_mean=0.5,
  )


class ImageSuperResolutionDataset(data.Dataset):

  def __init__(self, mode, params, lr_files, hr_files):
    super(ImageSuperResolutionDataset, self).__init__()
    self.mode = mode
    self.params = params
    self.lr_files = lr_files
    self.hr_files = hr_files

  def __getitem__(self, index):
    if self.mode == common.modes.PREDICT:
      lr_image = np.asarray(Image.open(self.lr_files[index][1]))
      lr_image = transforms.functional.to_tensor(lr_image)
      return lr_image, self.hr_files[index][0]

    if self.mode == common.modes.TRAIN:
      index = index // self.params.num_patches

    lr_image, hr_image = self._load_item(index)
    lr_image, hr_image = self._sample_patch(lr_image, hr_image)
    lr_image, hr_image = self._augment(lr_image, hr_image)

    if self.mode == common.modes.EVAL:
      lr_image = Image.fromarray(np.ascontiguousarray(lr_image))
      lr_image = ImageOps.pad(lr_image, (int(max(hr_image.shape)/self.params.scale), int(max(hr_image.shape)/self.params.scale)), method=Image.ANTIALIAS, centering=(0.5, 0.5), color=0)

    lr_image = np.ascontiguousarray(lr_image)
    hr_image = np.ascontiguousarray(hr_image)
    lr_image = transforms.functional.to_tensor(lr_image)
    hr_image = transforms.functional.to_tensor(hr_image)

    return lr_image, hr_image

  def _load_item(self, index):
    lr_image = np.asarray(Image.open(self.lr_files[index][1]))
    hr_image = np.asarray(Image.open(self.hr_files[index][1]))
    return lr_image, hr_image

  def _sample_patch(self, lr_image, hr_image):
    if self.mode == common.modes.TRAIN:
      # sample patch while training
      x = random.randrange(
          self.params.ignored_boundary_size, lr_image.shape[0] -
          self.params.lr_patch_size + 1 - self.params.ignored_boundary_size)
      y = random.randrange(
          self.params.ignored_boundary_size, lr_image.shape[1] -
          self.params.lr_patch_size + 1 - self.params.ignored_boundary_size)
      lr_image = lr_image[x:x + self.params.lr_patch_size, y:y +
                          self.params.lr_patch_size]
      hr_image = hr_image[x *
                          self.params.scale:(x + self.params.lr_patch_size) *
                          self.params.scale, y *
                          self.params.scale:(y + self.params.lr_patch_size) *
                          self.params.scale]
    else:
      hr_image = hr_image[:lr_image.shape[0] *
                          self.params.scale, :lr_image.shape[1] *
                          self.params.scale]
    return lr_image, hr_image

  def _augment(self, lr_image, hr_image):
    if self.mode == common.modes.TRAIN:
      # augmentation while training
      if random.random() < 0.5:
        lr_image = lr_image[::-1]
        hr_image = hr_image[::-1]
      if random.random() < 0.5:
        lr_image = lr_image[:, ::-1]
        hr_image = hr_image[:, ::-1]
      if random.random() < 0.5:
        lr_image = np.swapaxes(lr_image, 0, 1)
        hr_image = np.swapaxes(hr_image, 0, 1)
    return lr_image, hr_image

  def __len__(self):
    if self.mode == common.modes.TRAIN:
      return len(self.lr_files) * self.params.num_patches
    else:
      return len(self.lr_files)


class ImageSuperResolutionHdf5Dataset(ImageSuperResolutionDataset):

  def __init__(
      self,
      mode,
      params,
      lr_files,
      hr_files,
      lr_cache_file,
      hr_cache_file,
      lib_hdf5='h5py',
  ):
    super(ImageSuperResolutionHdf5Dataset, self).__init__(
        mode,
        params,
        lr_files,
        hr_files,
    )
    self.lr_cache_file = common.io.Hdf5(lr_cache_file, lib_hdf5)
    self.hr_cache_file = common.io.Hdf5(hr_cache_file, lib_hdf5)

    cache_dir = os.path.dirname(lr_cache_file)
    if not os.path.exists(cache_dir):
      os.makedirs(cache_dir)

    if not os.path.exists(lr_cache_file):
      for lr_file in self.lr_files:
        self.lr_cache_file.add(lr_file[0], np.asarray(Image.open(lr_file[1])))
    if self.mode != common.modes.PREDICT:
      if not os.path.exists(hr_cache_file):
        for hr_file in self.hr_files:
          self.hr_cache_file.add(hr_file[0], np.asarray(Image.open(hr_file[1])))

  def _load_item(self, index):
    lr_image = self.lr_cache_file.get(self.lr_files[index][0])
    hr_image = self.hr_cache_file.get(self.hr_files[index][0])
    return lr_image, hr_image


class ImageSuperResolutionBicubicDataset(ImageSuperResolutionDataset):

  def __init__(self, mode, params, hr_files):
    super(ImageSuperResolutionBicubicDataset, self).__init__(
        mode,
        params,
        hr_files,
        hr_files,
    )

  def __getitem__(self, index):
    if self.mode == common.modes.PREDICT:
      hr_image = np.asarray(Image.open(self.lr_files[index][1]))
      if hr_image.shape[0] % self.params.scale:
        hr_image = hr_image[:-(hr_image.shape[0] % self.params.scale), :]
      if hr_image.shape[1] % self.params.scale:
        hr_image = hr_image[:, :-(hr_image.shape[1] % self.params.scale)]
      lr_image = imresize(hr_image, scalar_scale=1 / self.params.scale)
      lr_image = transforms.functional.to_tensor(lr_image)
      return lr_image, self.hr_files[index][0]
    else:
      return super(ImageSuperResolutionBicubicDataset, self).__getitem__(index)

  def _load_item(self, index):
    hr_image = np.asarray(Image.open(self.hr_files[index][1]))
    lr_image = hr_image
    return lr_image, hr_image

  def _sample_patch(self, lr_image, hr_image):
    if self.mode == common.modes.TRAIN:
      # sample patch while training
      hr_ignored_boundary_size = self.params.ignored_boundary_size * self.params.scale
      hr_patch_size = self.params.lr_patch_size * self.params.scale + hr_ignored_boundary_size * 2
      try:
        x = random.randrange(0, hr_image.shape[0] - hr_patch_size + 1)
        y = random.randrange(0, hr_image.shape[1] - hr_patch_size + 1)
      except ValueError:
        print(hr_image, hr_patch_size)
      hr_image = hr_image[x:x + hr_patch_size, y:y + hr_patch_size]
      lr_image = imresize(hr_image, scalar_scale=1 / self.params.scale)
      lr_image = lr_image[
          self.params.ignored_boundary_size:-self.params.ignored_boundary_size,
          self.params.ignored_boundary_size:-self.params.ignored_boundary_size]
      hr_image = hr_image[hr_ignored_boundary_size:-hr_ignored_boundary_size,
                          hr_ignored_boundary_size:-hr_ignored_boundary_size]
    else:
      if hr_image.shape[0] % self.params.scale:
        hr_image = hr_image[:-(hr_image.shape[0] % self.params.scale), :]
      if hr_image.shape[1] % self.params.scale:
        hr_image = hr_image[:, :-(hr_image.shape[1] % self.params.scale)]
      hr_image = np.asarray(hr_image)
      lr_image = imresize(hr_image, scalar_scale=1 / self.params.scale)
    return lr_image, hr_image


class ImageSuperResolutionBicubicHdf5Dataset(ImageSuperResolutionBicubicDataset
                                            ):

  def __init__(
      self,
      mode,
      params,
      hr_files,
      hr_cache_file,
      lib_hdf5='h5py',
  ):
    super(ImageSuperResolutionBicubicHdf5Dataset, self).__init__(
        mode,
        params,
        hr_files,
    )
    self.hr_cache_file = common.io.Hdf5(hr_cache_file, lib_hdf5)

    cache_dir = os.path.dirname(hr_cache_file)
    if not os.path.exists(cache_dir):
      os.makedirs(cache_dir)

    if self.mode != common.modes.PREDICT:
      if not os.path.exists(hr_cache_file):
        for hr_file in self.hr_files:
          self.hr_cache_file.add(hr_file[0], np.asarray(Image.open(hr_file[1])))

  def _load_item(self, index):
    hr_image = self.hr_cache_file.get(self.hr_files[index][0])
    lr_image = hr_image
    return lr_image, hr_image
