# Copyright 2021 Huawei Technologies Co., Ltd
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
# limitations under the License.from __future__ import print_function

from .PreProcessImage import PreProcessIm
from .Prefetcher import Prefetcher
import numpy as np


class Dataset(object):
  """The core elements of a dataset.    
  Args:
    final_batch: bool. The last batch may not be complete, if to abandon this 
      batch, set 'final_batch' to False.
  """

  def __init__(
      self,
      dataset_size=None,
      batch_size=None,
      final_batch=True,
      shuffle=True,
      num_prefetch_threads=1,
      prng=np.random,
      **pre_process_im_kwargs):

    self.pre_process_im = PreProcessIm(
      prng=prng,
      **pre_process_im_kwargs)

    self.prefetcher = Prefetcher(
      self.get_sample,
      dataset_size,
      batch_size,
      final_batch=final_batch,
      num_threads=num_prefetch_threads)

    self.shuffle = shuffle
    self.epoch_done = True
    self.prng = prng

  def set_mirror_type(self, mirror_type):
    self.pre_process_im.set_mirror_type(mirror_type)

  def get_sample(self, ptr):
    """Get one sample to put to queue."""
    raise NotImplementedError

  def next_batch(self):
    """Get a batch from the queue."""
    raise NotImplementedError

  def set_batch_size(self, batch_size):
    """You can change batch size, had better at the beginning of a new epoch.
    """
    self.prefetcher.set_batch_size(batch_size)
    self.epoch_done = True

  def stop_prefetching_threads(self):
    """This can be called to stop threads, e.g. after finishing using the 
    dataset, or when existing the python main program."""
    self.prefetcher.stop()
