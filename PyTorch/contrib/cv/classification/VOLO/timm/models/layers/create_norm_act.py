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
""" NormAct (Normalizaiton + Activation Layer) Factory

Create norm + act combo modules that attempt to be backwards compatible with separate norm + act
isntances in models. Where these are used it will be possible to swap separate BN + act layers with
combined modules like IABN or EvoNorms.

Hacked together by / Copyright 2020 Ross Wightman
"""
import types
import functools

import torch
import torch.nn as nn

from .evo_norm import EvoNormBatch2d, EvoNormSample2d
from .norm_act import BatchNormAct2d, GroupNormAct
from .inplace_abn import InplaceAbn

_NORM_ACT_TYPES = {BatchNormAct2d, GroupNormAct, EvoNormBatch2d, EvoNormSample2d, InplaceAbn}
_NORM_ACT_REQUIRES_ARG = {BatchNormAct2d, GroupNormAct, InplaceAbn}  # requires act_layer arg to define act type


def get_norm_act_layer(layer_class):
    layer_class = layer_class.replace('_', '').lower()
    if layer_class.startswith("batchnorm"):
        layer = BatchNormAct2d
    elif layer_class.startswith("groupnorm"):
        layer = GroupNormAct
    elif layer_class == "evonormbatch":
        layer = EvoNormBatch2d
    elif layer_class == "evonormsample":
        layer = EvoNormSample2d
    elif layer_class == "iabn" or layer_class == "inplaceabn":
        layer = InplaceAbn
    else:
        assert False, "Invalid norm_act layer (%s)" % layer_class
    return layer


def create_norm_act(layer_type, num_features, apply_act=True, jit=False, **kwargs):
    layer_parts = layer_type.split('-')  # e.g. batchnorm-leaky_relu
    assert len(layer_parts) in (1, 2)
    layer = get_norm_act_layer(layer_parts[0])
    #activation_class = layer_parts[1].lower() if len(layer_parts) > 1 else ''   # FIXME support string act selection?
    layer_instance = layer(num_features, apply_act=apply_act, **kwargs)
    if jit:
        layer_instance = torch.jit.script(layer_instance)
    return layer_instance


def convert_norm_act(norm_layer, act_layer):
    assert isinstance(norm_layer, (type, str,  types.FunctionType, functools.partial))
    assert act_layer is None or isinstance(act_layer, (type, str, types.FunctionType, functools.partial))
    norm_act_kwargs = {}

    # unbind partial fn, so args can be rebound later
    if isinstance(norm_layer, functools.partial):
        norm_act_kwargs.update(norm_layer.keywords)
        norm_layer = norm_layer.func

    if isinstance(norm_layer, str):
        norm_act_layer = get_norm_act_layer(norm_layer)
    elif norm_layer in _NORM_ACT_TYPES:
        norm_act_layer = norm_layer
    elif isinstance(norm_layer,  types.FunctionType):
        # if function type, must be a lambda/fn that creates a norm_act layer
        norm_act_layer = norm_layer
    else:
        type_name = norm_layer.__name__.lower()
        if type_name.startswith('batchnorm'):
            norm_act_layer = BatchNormAct2d
        elif type_name.startswith('groupnorm'):
            norm_act_layer = GroupNormAct
        else:
            assert False, f"No equivalent norm_act layer for {type_name}"

    if norm_act_layer in _NORM_ACT_REQUIRES_ARG:
        # pass `act_layer` through for backwards compat where `act_layer=None` implies no activation.
        # In the future, may force use of `apply_act` with `act_layer` arg bound to relevant NormAct types
        norm_act_kwargs.setdefault('act_layer', act_layer)
    if norm_act_kwargs:
        norm_act_layer = functools.partial(norm_act_layer, **norm_act_kwargs)  # bind/rebind args
    return norm_act_layer
