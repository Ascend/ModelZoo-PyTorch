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
from collections import OrderedDict
import torch.nn as nn


class ModelBook:
    """Maintain the mapping between modules and their paths.

    Example:
        book = ModelBook(model_ft)
        for p, m in book.conv2d_modules():
            print('path:', p, 'num of filters:', m.out_channels)
            assert m is book.get_module(p)
    """

    def __init__(self, model):
        self._model = model
        self._modules = OrderedDict()
        self._paths = OrderedDict()
        path = []
        self._construct(self._model, path)

    def _construct(self, module, path):
        if not module._modules:
            return
        for name, m in module._modules.items():
            cur_path = tuple(path + [name])
            self._paths[m] = cur_path
            self._modules[cur_path] = m
            self._construct(m, path + [name])

    def conv2d_modules(self):
        return self.modules(nn.Conv2d)

    def linear_modules(self):
        return self.modules(nn.Linear)

    def modules(self, module_type=None):
        for p, m in self._modules.items():
            if not module_type or isinstance(m, module_type):
                yield p, m

    def num_of_conv2d_modules(self):
        return self.num_of_modules(nn.Conv2d)

    def num_of_conv2d_filters(self):
        """Return the sum of out_channels of all conv2d layers.

        Here we treat the sub weight with size of [in_channels, h, w] as a single filter.
        """
        num_filters = 0
        for _, m in self.conv2d_modules():
            num_filters += m.out_channels
        return num_filters

    def num_of_linear_modules(self):
        return self.num_of_modules(nn.Linear)

    def num_of_linear_filters(self):
        num_filters = 0
        for _, m in self.linear_modules():
            num_filters += m.out_features
        return num_filters

    def num_of_modules(self, module_type=None):
        num = 0
        for p, m in self._modules.items():
            if not module_type or isinstance(m, module_type):
                num += 1
        return num

    def get_module(self, path):
        return self._modules.get(path)

    def get_path(self, module):
        return self._paths.get(module)

    def update(self, path, module):
        old_module = self._modules[path]
        del self._paths[old_module]
        self._paths[module] = path
        self._modules[path] = module
