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
import os

import common.modes
import datasets._vsr

LOCAL_DIR = 'data/REDS/'
TRAIN_DIR = LOCAL_DIR + 'train/'
EVAL_DIR = LOCAL_DIR + 'val/'


def update_argparser(parser):
  datasets._vsr.update_argparser(parser)
  parser.add_argument(
      '--reds_type',
      help='REDS dataset type.',
      choices=['blur', 'blur_comp', 'sharp_bicubic', 'sharp_bicubic'],
      default='blur',
      type=str)
  parser.add_argument(
      '--input_dir', help='Directory of input files in predict mode.')
  parser.set_defaults(
      num_channels=3,
      train_batch_size=32,
      eval_batch_size=2,
      num_patches=100,
      train_hr_dir=TRAIN_DIR + 'train_sharp/',
      eval_hr_dir=EVAL_DIR + 'val_sharp/',
  )
  args, _ = parser.parse_known_args()
  if args.reds_type == 'blur' or args.reds_type == 'blur_comp':
    parser.set_defaults(
        scale=1,
        train_lr_dir=TRAIN_DIR + 'train_{}/'.format(args.reds_type),
        eval_lr_dir=EVAL_DIR + 'val_{}/'.format(args.reds_type),
    )
  elif args.reds_type == 'sharp_bicubic' or args.reds_type == 'blur_bicubic':
    parser.set_defaults(
        scale=4,
        train_lr_dir=TRAIN_DIR + 'train_{}/X{}/'.format(args.reds_type, 4),
        eval_lr_dir=EVAL_DIR + 'val_{}/X{}/'.format(args.reds_type, 4),
    )
  else:
    raise NotImplementedError('Needs to tune hyper parameters for new dataset.')


def get_dataset(mode, params):
  if mode == common.modes.TRAIN:
    return REDS(mode, params)
  elif mode == common.modes.EVAL:
    return REDS_9(mode, params)
  else:
    return REDS_(mode, params)


class REDS(datasets._vsr.VideoSuperResolutionHDF5Dataset):

  def __init__(self, mode, params):
    lr_cache_file = 'cache/reds_{}_{}.h5'.format(mode, params.reds_type)
    hr_cache_file = 'cache/reds_{}_sharp.h5'.format(mode)

    lr_dir = {
        common.modes.TRAIN: params.train_lr_dir,
        common.modes.EVAL: params.eval_lr_dir,
        common.modes.PREDICT: '',
    }[mode]
    hr_dir = {
        common.modes.TRAIN: params.train_hr_dir,
        common.modes.EVAL: params.eval_hr_dir,
        common.modes.PREDICT: '',
    }[mode]

    lr_files = list_video_files(lr_dir)
    if mode == common.modes.PREDICT:
      hr_files = lr_files
    else:
      hr_files = list_video_files(hr_dir)

    super(REDS, self).__init__(
        mode,
        params,
        lr_files,
        hr_files,
        lr_cache_file,
        hr_cache_file,
    )


class REDS_9(datasets._vsr.VideoSuperResolutionDataset):

  def __init__(self, mode, params):
    lr_dir = {
        common.modes.TRAIN: params.train_lr_dir,
        common.modes.EVAL: params.eval_lr_dir,
        common.modes.PREDICT: '',
    }[mode]
    hr_dir = {
        common.modes.TRAIN: params.train_hr_dir,
        common.modes.EVAL: params.eval_hr_dir,
        common.modes.PREDICT: '',
    }[mode]

    lr_files = list_video_files(lr_dir)
    if mode == common.modes.PREDICT:
      hr_files = lr_files
    else:
      hr_files = list_video_files(hr_dir)

    super(REDS_9, self).__init__(
        mode,
        params,
        lr_files,
        hr_files,
    )

  def __getitem__(self, index):
    return super().__getitem__(index * 10 + 9)

  def __len__(self):
    return super().__len__() // 10


class REDS_(datasets._vsr.VideoSuperResolutionDataset):

  def __init__(self, mode, params):
    lr_dir = {
        common.modes.TRAIN: params.train_lr_dir,
        common.modes.EVAL: params.eval_lr_dir,
        common.modes.PREDICT: params.input_dir,
    }[mode]

    lr_files = list_video_files(lr_dir)
    if mode == common.modes.PREDICT:
      hr_files = lr_files

    super(REDS_, self).__init__(
        mode,
        params,
        lr_files,
        hr_files,
    )


def list_video_files(d):
  video_files = []
  for video in sorted(os.listdir(d)):
    files = sorted(os.listdir(os.path.join(d, video)))
    files = [f for f in files if f.endswith(".png")]
    video_files.append((video, [
        (os.path.join(video, f), os.path.join(d, video, f)) for f in files
    ]))
  return video_files
