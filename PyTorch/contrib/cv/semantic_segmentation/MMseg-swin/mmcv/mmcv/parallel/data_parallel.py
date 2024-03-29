# -*- coding: utf-8 -*-
# BSD 3-Clause License
#
# Copyright (c) 2017
# All rights reserved.
# Copyright 2022 Huawei Technologies Co., Ltd
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
# ==========================================================================

# -*- coding: utf-8 -*-
# BSD 3-Clause License
#
# Copyright (c) 2017
# All rights reserved.
# Copyright 2022 Huawei Technologies Co., Ltd
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
# ==========================================================================

# Copyright (c) OpenMMLab. All rights reserved.
from itertools import chain
from typing import List, Tuple

from torch.nn.parallel import DataParallel

from .scatter_gather import ScatterInputs, scatter_kwargs


class MMDataParallel(DataParallel):
    """The DataParallel module that supports DataContainer.

    MMDataParallel has two main differences with PyTorch DataParallel:

    - It supports a custom type :class:`DataContainer` which allows more
      flexible control of input data during both GPU and CPU inference.
    - It implements two more APIs ``train_step()`` and ``val_step()``.

    .. warning::
        MMDataParallel only supports single GPU training, if you need to
        train with multiple GPUs, please use MMDistributedDataParallel
        instead. If you have multiple GPUs and you just want to use
        MMDataParallel, you can set the environment variable
        ``CUDA_VISIBLE_DEVICES=0`` or instantiate ``MMDataParallel`` with
        ``device_ids=[0]``.

    Args:
        module (:class:`nn.Module`): Module to be encapsulated.
        device_ids (list[int]): Device IDS of modules to be scattered to.
            Defaults to None when GPU is not available.
        output_device (str | int): Device ID for output. Defaults to None.
        dim (int): Dimension used to scatter the data. Defaults to 0.
    """

    def __init__(self, *args, dim: int = 0, **kwargs):
        super().__init__(*args, dim=dim, **kwargs)
        self.dim = dim

    def forward(self, *inputs, **kwargs):
        """Override the original forward function.

        The main difference lies in the CPU inference where the data in
        :class:`DataContainers` will still be gathered.
        """
        if not self.device_ids:
            # We add the following line thus the module could gather and
            # convert data containers as those in GPU inference
            inputs, kwargs = self.scatter(inputs, kwargs, [-1])
            return self.module(*inputs[0], **kwargs[0])
        else:
            return super().forward(*inputs, **kwargs)

    def forward_val(self, *inputs, **kwargs):
        """Override the original forward function.

        The main difference lies in the CPU inference where the datas in
        :class:`DataContainers` will still be gathered.
        """
        if not self.device_ids:
            # We add the following line thus the module could gather and
            # convert data containers as those in GPU inference
            inputs, kwargs = self.scatter(inputs, kwargs, [-1])
            return self.module(*inputs[0], **kwargs[0])
        else:
            return super().forward(*inputs, **kwargs)

    def scatter(self, inputs: ScatterInputs, kwargs: ScatterInputs,
                device_ids: List[int]) -> Tuple[tuple, tuple]:
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def train_step(self, *inputs, **kwargs):
        if not self.device_ids:
            # We add the following line thus the module could gather and
            # convert data containers as those in GPU inference
            inputs, kwargs = self.scatter(inputs, kwargs, [-1])
            return self.module.train_step(*inputs[0], **kwargs[0])

        assert len(self.device_ids) == 1, \
            ('MMDataParallel only supports single GPU training, if you need to'
             ' train with multiple GPUs, please use MMDistributedDataParallel'
             ' instead.')

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError(
                    'module must have its parameters and buffers '
                    f'on device {self.src_device_obj} (device_ids[0]) but '
                    f'found one of them on device: {t.device}')

        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        return self.module.train_step(*inputs[0], **kwargs[0])

    def val_step(self, *inputs, **kwargs):
        if not self.device_ids:
            # We add the following line thus the module could gather and
            # convert data containers as those in GPU inference
            inputs, kwargs = self.scatter(inputs, kwargs, [-1])
            return self.module.val_step(*inputs[0], **kwargs[0])

        assert len(self.device_ids) == 1, \
            ('MMDataParallel only supports single GPU training, if you need to'
             ' train with multiple GPUs, please use MMDistributedDataParallel'
             ' instead.')

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError(
                    'module must have its parameters and buffers '
                    f'on device {self.src_device_obj} (device_ids[0]) but '
                    f'found one of them on device: {t.device}')

        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        return self.module.val_step(*inputs[0], **kwargs[0])
