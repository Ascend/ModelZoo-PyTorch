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
"""File IO helper."""

import warnings


class Hdf5:

  def __init__(self, fname, lib='h5py'):
    self.fname = fname
    self.lib = lib
    self.file = None

  def add(self, key, value):
    if self.lib == 'h5py':
      import h5py
      with h5py.File(self.fname, 'a', libver='latest') as f:
        f.create_dataset(
            key,
            data=value,
            maxshape=value.shape,
            compression='lzf',
            shuffle=True,
            track_times=False,
            #track_order=False,
        )
    elif self.lib == 'pytables':
      import tables
      filters = tables.Filters(complevel=8, complib='blosc', bitshuffle=True)
      original_warnings = list(warnings.filters)
      warnings.simplefilter('ignore', tables.NaturalNameWarning)
      with tables.File(self.fname, 'a', filters=filters) as f:
        f.create_carray(
            f.root,
            key,
            obj=value,
            track_times=False,
        )
      warnings.filters = original_warnings
    else:
      raise NotImplementedError

  def get(self, key):
    if self.lib == 'h5py':
      if not self.file:
        import h5py
        self.file = h5py.File(self.fname, 'r', libver='latest')
      return self.file[key]
    elif self.lib == 'pytables':
      if not self.file:
        import tables
        self.file = tables.File(self.fname, 'r')
      return self.file.root[key]
    else:
      raise NotImplementedError
