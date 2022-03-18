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
# Copyright (c) Runpei Dong, ArChip Lab.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
import torch
from torch import nn

def exists(val):
    return val is not None

class Extractor(nn.Module):
    def __init__(
        self,
        vit,
        device = None,
        layer_name = 'transformer',
        layer_save_input = False,
        return_embeddings_only = False
    ):
        super().__init__()
        self.vit = vit

        self.data = None
        self.latents = None
        self.hooks = []
        self.hook_registered = False
        self.ejected = False
        self.device = device

        self.layer_name = layer_name
        self.layer_save_input = layer_save_input # whether to save input or output of layer
        self.return_embeddings_only = return_embeddings_only

    def _hook(self, _, inputs, output):
        tensor_to_save = inputs if self.layer_save_input else output
        self.latents = tensor_to_save.clone().detach()

    def _register_hook(self):
        assert hasattr(self.vit, self.layer_name), 'layer whose output to take as embedding not found in vision transformer'
        layer = getattr(self.vit, self.layer_name)
        handle = layer.register_forward_hook(self._hook)
        self.hooks.append(handle)
        self.hook_registered = True

    def eject(self):
        self.ejected = True
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        return self.vit

    def clear(self):
        del self.latents
        self.latents = None

    def forward(
        self,
        img,
        return_embeddings_only = False
    ):
        assert not self.ejected, 'extractor has been ejected, cannot be used anymore'
        self.clear()
        if not self.hook_registered:
            self._register_hook()

        pred = self.vit(img)

        target_device = self.device if exists(self.device) else img.device
        latents = self.latents.to(target_device)

        if return_embeddings_only or self.return_embeddings_only:
            return latents

        return pred, latents
