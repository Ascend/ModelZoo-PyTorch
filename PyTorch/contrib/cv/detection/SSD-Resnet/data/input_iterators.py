# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
# from SSD import _C as C


class ConvertDaliInputIterator(object):
    def __init__(self, dali_it):
        self._dali_it = dali_it

    def __next__(self):
        batch = self._dali_it.__next__()
        return batch[0]['image'], batch[0]['bbox'], batch[0]['label']

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def reset(self):
        return None


# Abstraction over EncodingInputIterator to allow the input pipeline to run at
# a larger batch size than the training pipeline
class RateMatcher(object):
    def __init__(self, input_it, output_size):
        self._input_it = input_it
        self._output_size = output_size

        self._img = None
        self._bbox = None
        self._label = None
        self._offset_offset = 0

    def __next__(self):
        if (self._img is None) or (self._offset_offset >= len(self._img)):
            self._offset_offset = 0
            (self._img, self._bbox, self._label) = self._input_it.__next__()
            if self._img is not None and len(self._img) == 0:
                self._img = None
            if self._img is None:
                return (None, None, None)
            # make sure all three tensors are same size
            assert (len(self._img) == len(self._bbox)) and (len(self._img) == len(self._label))

            # semantics of split() are that it will make as many chunks as
            # necessary with all chunks of output_size except possibly the last
            # if the input size is not perfectly divisble by the the output_size
            self._img   = self._img.split(self._output_size, dim=0)
            self._bbox  = self._bbox.split(self._output_size, dim=0)
            self._label = self._label.split(self._output_size, dim=0)

        output = (self._img[self._offset_offset],
                  self._bbox[self._offset_offset],
                  self._label[self._offset_offset])
        self._offset_offset = self._offset_offset + 1
        return output

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def reset(self):
        self._img = None
        self._bbox = None
        self._label = None
        self._offset_offset = 0

        return self._input_it.reset()


class FakeInputIterator(object):
    def __init__(self, input_it, epoch_size, n_gpus=1):
        self._input_it    = input_it
        self._epoch_size  = epoch_size
        self._n_gpus      = n_gpus
        self._saved_batch = None
        self._frame_count = 0   # current number of images into the epoch we've fetched
        self._input_size  = None

    def __next__(self):
        if self._saved_batch is None:
            # first iter only: really fetch a batch from input_it
            self._saved_batch = self._input_it.__next__()
            # saved batch is a 3-tuple (img, bbox, label) and
            # it's "size" is the len of the img tensor
            self._input_size = len(self._saved_batch[0])
            self._frame_count = 0

        if self._frame_count >= self._epoch_size:
            # remember how many we've fetched some from the next epoch
            self._frame_count = self._frame_count % self._epoch_size
            raise StopIteration

        self._frame_count = self._frame_count + (self._input_size * self._n_gpus)
        return self._saved_batch

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def reset(self):
        return None
