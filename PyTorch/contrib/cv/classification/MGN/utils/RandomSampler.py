"""
BSD 3-Clause License

Copyright (c) Soumith Chintala 2016,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Copyright 2020 Huawei Technologies Co., Ltd

Licensed under the BSD 3-Clause License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://spdx.org/licenses/BSD-3-Clause.html

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import random
import collections
from torch.utils.data import sampler
from torch.utils.data.distributed import DistributedSampler

# class RandomSampler(sampler.Sampler):
#     def __init__(self, data_source, batch_id, batch_image):
#         super(RandomSampler, self).__init__(data_source)

#         self.data_source = data_source
#         self.batch_image = batch_image
#         self.batch_id = batch_id

#         self._id2index = collections.defaultdict(list)
#         for idx, path in enumerate(data_source.imgs):
#             _id = data_source.id(path)
#             self._id2index[_id].append(idx)

#     def __iter__(self):
#         unique_ids = self.data_source.unique_ids
#         random.shuffle(unique_ids)

#         imgs = []
#         for _id in unique_ids:
#             imgs.extend(self._sample(self._id2index[_id], self.batch_image))
#         return iter(imgs)

#     def __len__(self):
#         return len(self._id2index) * self.batch_image

#     @staticmethod
#     def _sample(population, k):
#         if len(population) < k:
#             population = population * k
#         return random.sample(population, k)

class RandomSamplerDDP(DistributedSampler):
    def __init__(self, data_source, batch_id, batch_image):
        super(RandomSamplerDDP, self).__init__(data_source)

        self.data_source = data_source
        self.batch_image = batch_image
        self.batch_id = batch_id

        self._id2index = collections.defaultdict(list)
        for idx, path in enumerate(data_source.imgs):
            _id = data_source.id(path)
            self._id2index[_id].append(idx)

    def __iter__(self):
        unique_ids = self.data_source.unique_ids
        random.shuffle(unique_ids)

        imgs = []
        for _id in unique_ids:
            imgs.extend(self._sample(self._id2index[_id], self.batch_image))
        return iter(imgs)

    def __len__(self):
        return len(self._id2index) * self.batch_image

    @staticmethod
    def _sample(population, k):
        if len(population) < k:
            population = population * k
        return random.sample(population, k)



class RandomSampler(sampler.Sampler):
    def __init__(self, data_source, batch_id, batch_image,rank=0,world_size=1):
        super(RandomSampler, self).__init__(data_source)

        self.data_source = data_source
        self.batch_image = batch_image
        self.batch_id = batch_id
        self.rank = rank
        self.world_size = world_size

        self._id2index = collections.defaultdict(list)
        for idx, path in enumerate(data_source.imgs):
            _id = data_source.id(path)
            self._id2index[_id].append(idx)

    def __iter__(self):
        unique_ids = self.data_source.unique_ids
        random.shuffle(unique_ids)

        imgs = []
        for _id in unique_ids:
            imgs.extend(self._sample(self._id2index[_id], self.batch_image))
        length =len(imgs)
        # imgs = imgs[self.rank : length: self.world_size]
        if length % self.world_size != 0:
            append_num = self.world_size - length % self.world_size
            imgs.extend(imgs[:append_num])
        length =len(imgs)
        assert length % self.world_size == 0
        eve_len = length // self.world_size
        imgs = imgs[eve_len * self.rank: eve_len * (self.rank+1)]

        return iter(imgs)

    def __len__(self):
        return int(len(self._id2index) * self.batch_image / self.world_size)

    @staticmethod
    def _sample(population, k):
        if len(population) < k:
            population = population * k
        return random.sample(population, k)
