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

import time
import inspect
import subprocess
from contextlib import contextmanager

import torch


def int_tuple(s):
  return tuple(int(i) for i in s.split(','))


def float_tuple(s):
  return tuple(float(i) for i in s.split(','))


def str_tuple(s):
  return tuple(s.split(','))


def bool_flag(s):
  if s == '1':
    return True
  elif s == '0':
    return False
  msg = 'Invalid value "%s" for bool flag (should be 0 or 1)'
  raise ValueError(msg % s)


def lineno():
  return inspect.currentframe().f_back.f_lineno


def get_gpu_memory():
  torch.npu.synchronize()
  opts = [
      'nvidia-smi', '-q', '--gpu=' + str(0), '|', 'grep', '"Used GPU Memory"'
  ]
  cmd = str.join(' ', opts)
  ps = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
  output = ps.communicate()[0].decode('utf-8')
  output = output.split("\n")[1].split(":")
  consumed_mem = int(output[1].strip().split(" ")[0])
  return consumed_mem


@contextmanager
def timeit(msg, should_time=True):
  if should_time:
    torch.npu.synchronize()
    t0 = time.time()
  yield
  if should_time:
    torch.npu.synchronize()
    t1 = time.time()
    duration = (t1 - t0) * 1000.0
    print('%s: %.2f ms' % (msg, duration))


class LossManager(object):
  def __init__(self):
    self.total_loss = None
    self.all_losses = {}

  def add_loss(self, loss, name, weight=1.0):
    cur_loss = loss * weight
    if self.total_loss is not None:
      self.total_loss += cur_loss
    else:
      self.total_loss = cur_loss

    self.all_losses[name] = cur_loss.data.cpu().item()

  def items(self):
    return self.all_losses.items()

