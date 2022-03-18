# -*- coding: utf-8 -*-
# Copyright (c) Soumith Chintala 2016,
# All rights reserved
#
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://spdx.org/licenses/BSD-3-Clause.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License


import torch.utils.data as tordata
import random


class TripletSampler(tordata.sampler.Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        while (True):
            sample_indices = list()
            pid_list = random.sample(
                list(self.dataset.label_set),
                self.batch_size[0])
            for pid in pid_list:
                _index = self.dataset.index_dict.loc[pid, :, :].values
                _index = _index[_index > 0].flatten().tolist()
                _index = random.choices(
                    _index,
                    k=self.batch_size[1])
                sample_indices += _index
            yield sample_indices

    def __len__(self):
        return self.dataset.data_size
