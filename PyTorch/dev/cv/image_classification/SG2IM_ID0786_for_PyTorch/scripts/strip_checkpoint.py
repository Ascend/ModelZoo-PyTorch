#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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

import argparse, os
import torch


"""
Checkpoints saved by train.py contain not only model parameters but also
optimizer states, losses, a history of generated images, and other statistics.
This information is very useful for development and debugging models, but makes
the saved checkpoints very large. This utility script strips away all extra
information from saved checkpoints, keeping only the saved models.
"""


parser = argparse.ArgumentParser()
parser.add_argument('--input_checkpoint', default=None)
parser.add_argument('--output_checkpoint', default=None)
parser.add_argument('--input_dir', default=None)
parser.add_argument('--output_dir', default=None)
parser.add_argument('--keep_discriminators', type=int, default=1)


def main(args):
  if args.input_checkpoint is not None:
    handle_checkpoint(args, args.input_checkpoint, args.output_checkpoint)
  if args.input_dir is not None:
    handle_dir(args, args.input_dir, args.output_dir)


def handle_dir(args, input_dir, output_dir):
  for fn in os.listdir(input_dir):
    if not fn.endswith('.pt'):
      continue
    input_path = os.path.join(input_dir, fn)
    output_path = os.path.join(output_dir, fn)
    handle_checkpoint(args, input_path, output_path)


def handle_checkpoint(args, input_path, output_path):
  input_checkpoint = torch.load(input_path)
  keep = ['args', 'model_state', 'model_kwargs']
  if args.keep_discriminators == 1:
    keep += ['d_img_state', 'd_img_kwargs', 'd_obj_state', 'd_obj_kwargs']
  output_checkpoint = {}
  for k, v in input_checkpoint.items():
    if k in keep:
      output_checkpoint[k] = v
  torch.save(output_checkpoint, output_path)


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)

