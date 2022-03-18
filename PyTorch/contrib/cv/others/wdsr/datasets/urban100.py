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
import datasets._isr

LOCAL_DIR = 'data/Urban100/'


def update_argparser(parser):
  datasets._isr.update_argparser(parser)
  parser.add_argument(
      '--input_dir', help='Directory of input files in predict mode.')
  parser.set_defaults(
      num_channels=3,
      eval_batch_size=1,
  )


def get_dataset(mode, params):
  if mode == common.modes.EVAL:
    return Set_(mode, params)
  else:
    raise NotImplementedError


class Set_(datasets._isr.ImageSuperResolutionBicubicDataset):

  def __init__(self, mode, params):
    hr_dir = {
        common.modes.TRAIN: LOCAL_DIR,
        common.modes.EVAL: LOCAL_DIR,
        common.modes.PREDICT: params.input_dir,
    }[mode]

    hr_files = list_image_files(hr_dir)

    super(Set_, self).__init__(
        mode,
        params,
        hr_files,
    )


def list_image_files(d):
  files = sorted(os.listdir(d))
  files = [(f, os.path.join(d, f)) for f in files if f.endswith('.png')]
  return files
