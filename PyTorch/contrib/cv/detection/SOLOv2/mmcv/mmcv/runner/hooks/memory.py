# Copyright 2021 Huawei Technologies Co., Ltd
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

# Copyright (c) Open-MMLab. All rights reserved.
import torch

from .hook import Hook


class EmptyCacheHook(Hook):

    def __init__(self, before_epoch=False, after_epoch=True, after_iter=False):
        self._before_epoch = before_epoch
        self._after_epoch = after_epoch
        self._after_iter = after_iter

    def after_iter(self, runner):
        if self._after_iter:
            torch.cuda.empty_cache()

    def before_epoch(self, runner):
        if self._before_epoch:
            torch.cuda.empty_cache()

    def after_epoch(self, runner):
        if self._after_epoch:
            torch.cuda.empty_cache()
