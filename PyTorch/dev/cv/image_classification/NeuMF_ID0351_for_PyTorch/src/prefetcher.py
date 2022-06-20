# Copyright (c) 2020 Huawei Technologies Co., Ltd
# Copyright (c) 2019, Facebook CORPORATION.
# All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch


class Prefetcher(object):
    """Prefetcher using on npu device.

    Origin Code URL:
    https://github.com/implus/PytorchInsight/blob/master/classification/imagenet_fast.py#L280

    Args:
        loder (torch.utils.data.DataLoader or DataLoader like iterator):
            Using to generate inputs after preprocessing.
        stream (torch.npu.Stream): Default None.
            Because of the limitation of NPU's memory mechanism,
            if prefetcher is initialized repeatedly during training,
            a defined stream should be introduced to prevent memory leakage;
            if prefetcher is initialized only once during training,
            a defined stream is not necessary.

    Returns:
        float: tensors of shape (k, 5) and (k, 1). Labels are 0-based.
    """

    def __init__(self, loader, stream=None):
        self.loader = iter(loader)
        self.stream = stream if stream is not None else torch.npu.Stream()
        self.preload()

    def preload(self):
        try:
            self.user, self.item, self.rating = next(self.loader)
            assert isinstance(self.user, torch.IntTensor)
            self.rating = self.rating.float()
        except StopIteration:
            self.user = None
            self.item = None
            return

        with torch.npu.stream(self.stream):
            self.user = self.user.npu(non_blocking=True)
            self.item = self.item.npu(non_blocking=True)
            self.rating = self.rating.npu(non_blocking=True)

    def next(self):
        torch.npu.current_stream().wait_stream(self.stream)
        user = self.user
        item = self.item
        rating = self.rating
        if user is not None:
            self.preload()
        return user, item, rating
