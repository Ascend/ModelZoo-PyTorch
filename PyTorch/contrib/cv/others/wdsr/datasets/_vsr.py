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
"""Video Super-resolution dataset."""

import os
import random
from PIL import Image
import numpy as np

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import common.modes
import datasets._isr


def update_argparser(parser):
  datasets._isr.update_argparser(parser)
  parser.add_argument(
      '--train_temporal_size',
      help='Number of frames for training',
      default=5,
      type=int)
  parser.add_argument(
      '--eval_temporal_size',
      help='Number of frames for evaluation',
      default=5,
      type=int)
  parser.add_argument(
      '--train_temporal_padding_size',
      help='Number of frames for training',
      default=3,
      type=int)
  parser.add_argument(
      '--eval_temporal_padding_size',
      help='Number of frames for evaluation',
      default=3,
      type=int)
  parser.set_defaults(
      train_batch_size=16,
      eval_batch_size=1,
  )


class _SingleVideoSuperResolutionDataset(data.Dataset):

  def __init__(self, mode, params, video_name, lr_files, hr_files):
    super(_SingleVideoSuperResolutionDataset, self).__init__()
    self.mode = mode
    self.params = params
    self.video_name = video_name
    self.lr_files = lr_files
    self.hr_files = hr_files
    self.temporal_size = {
        common.modes.TRAIN: params.train_temporal_size,
        common.modes.EVAL: params.eval_temporal_size,
        common.modes.PREDICT: params.eval_temporal_size,
    }[mode]
    self.temporal_padding_size = {
        common.modes.TRAIN: params.train_temporal_padding_size,
        common.modes.EVAL: params.eval_temporal_padding_size,
        common.modes.PREDICT: params.eval_temporal_padding_size,
    }[mode]

  def __getitem__(self, index):
    t = index * self.temporal_size
    lr_files = [
        self.lr_files[min(len(self.lr_files) - 1, max(0, i))]
        for i in range(t - self.temporal_padding_size, t + self.temporal_size +
                       self.temporal_padding_size)
    ]
    hr_files = [self.hr_files[i] for i in range(t, t + self.temporal_size)]
    if self.mode == common.modes.PREDICT:
      lr_images = [
          transforms.functional.to_tensor(np.asarray(Image.open(lr_file[1])))
          for lr_file in lr_files
      ]
      lr_images = torch.stack(lr_images, dim=1)
      hr_files = [hr_file[0] for hr_file in hr_files]
      return lr_images, hr_files

    lr_images, hr_images = self._load_item(lr_files, hr_files)
    lr_images, hr_images = self._sample_patch(lr_images, hr_images)
    lr_images, hr_images = self._augment(lr_images, hr_images)

    lr_images = [np.ascontiguousarray(lr_image) for lr_image in lr_images]
    hr_images = [np.ascontiguousarray(hr_image) for hr_image in hr_images]
    lr_images = [
        transforms.functional.to_tensor(lr_image) for lr_image in lr_images
    ]
    hr_images = [
        transforms.functional.to_tensor(hr_image) for hr_image in hr_images
    ]
    lr_images = torch.stack(lr_images, dim=1)
    hr_images = torch.stack(hr_images, dim=1)

    return lr_images, hr_images

  def _load_item(self, lr_files, hr_files):
    lr_images = [np.asarray(Image.open(lr_file[1])) for lr_file in lr_files]
    hr_images = [np.asarray(Image.open(hr_file[1])) for hr_file in hr_files]
    return lr_images, hr_images

  def _sample_patch(self, lr_images, hr_images):
    if self.mode == common.modes.TRAIN:
      # sample patch while training
      x = random.randrange(
          self.params.ignored_boundary_size, lr_images[0].shape[0] -
          self.params.lr_patch_size + 1 - self.params.ignored_boundary_size)
      y = random.randrange(
          self.params.ignored_boundary_size, lr_images[0].shape[1] -
          self.params.lr_patch_size + 1 - self.params.ignored_boundary_size)
      lr_images = [
          lr_image[x:x + self.params.lr_patch_size, y:y +
                   self.params.lr_patch_size] for lr_image in lr_images
      ]
      hr_images = [
          hr_image[x * self.params.scale:(x + self.params.lr_patch_size) *
                   self.params.scale, y *
                   self.params.scale:(y + self.params.lr_patch_size) *
                   self.params.scale] for hr_image in hr_images
      ]
    return lr_images, hr_images

  def _augment(self, lr_images, hr_images):
    if self.mode == common.modes.TRAIN:
      # augmentation while training
      if random.random() < 0.5:
        lr_images = [lr_image[::-1] for lr_image in lr_images]
        hr_images = [hr_image[::-1] for hr_image in hr_images]
      if random.random() < 0.5:
        lr_images = [lr_image[:, ::-1] for lr_image in lr_images]
        hr_images = [hr_image[:, ::-1] for hr_image in hr_images]
      if random.random() < 0.5:
        lr_images = [np.swapaxes(lr_image, 0, 1) for lr_image in lr_images]
        hr_images = [np.swapaxes(hr_image, 0, 1) for hr_image in hr_images]
      if random.random() < 0.5:
        lr_images = reversed(lr_images)
        hr_images = reversed(hr_images)
    return lr_images, hr_images

  def __len__(self):
    if len(self.hr_files) % self.temporal_size:
      raise NotImplementedError
    return len(self.hr_files) // self.temporal_size


class VideoSuperResolutionDataset(data.ConcatDataset):

  def __init__(self, mode, params, lr_files, hr_files):
    video_datasets = []
    for (v, l), (_, h) in zip(lr_files, hr_files):
      video_datasets.append(
          _SingleVideoSuperResolutionDataset(mode, params, v, l, h))
    if mode == common.modes.TRAIN:
      video_datasets = video_datasets * params.num_patches
    super(VideoSuperResolutionDataset, self).__init__(video_datasets)


class _SingleVideoSuperResolutionHDF5Dataset(_SingleVideoSuperResolutionDataset
                                            ):

  def __init__(
      self,
      mode,
      params,
      video_name,
      lr_files,
      hr_files,
      lr_cache_file,
      hr_cache_file,
      lib_hdf5='h5py',
      init_hdf5=False,
  ):
    super(_SingleVideoSuperResolutionHDF5Dataset, self).__init__(
        mode,
        params,
        video_name,
        lr_files,
        hr_files,
    )
    self.lr_cache_file = common.io.Hdf5(lr_cache_file, lib_hdf5)
    self.hr_cache_file = common.io.Hdf5(hr_cache_file, lib_hdf5)

    if init_hdf5:
      cache_dir = os.path.dirname(lr_cache_file)
      if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

      for lr_file in self.lr_files:
        self.lr_cache_file.add(lr_file[0], np.asarray(Image.open(lr_file[1])))
      if self.mode != common.modes.PREDICT:
        for hr_file in self.hr_files:
          self.hr_cache_file.add(hr_file[0], np.asarray(Image.open(hr_file[1])))

  def _load_item(self, lr_files, hr_files):
    lr_images = [self.lr_cache_file.get(lr_file[0]) for lr_file in lr_files]
    hr_images = [self.hr_cache_file.get(hr_file[0]) for hr_file in hr_files]
    return lr_images, hr_images


class VideoSuperResolutionHDF5Dataset(data.ConcatDataset):

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
    video_datasets = []
    init_hdf5 = not os.path.exists(lr_cache_file)
    for (v, l), (_, h) in zip(lr_files, hr_files):
      video_datasets.append(
          _SingleVideoSuperResolutionHDF5Dataset(
              mode,
              params,
              v,
              l,
              h,
              lr_cache_file,
              hr_cache_file,
              lib_hdf5=lib_hdf5,
              init_hdf5=init_hdf5))
    if mode == common.modes.TRAIN:
      video_datasets = video_datasets * params.num_patches
    super(VideoSuperResolutionHDF5Dataset, self).__init__(video_datasets)
