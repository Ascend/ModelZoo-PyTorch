# Copyright 2022 Huawei Technologies Co., Ltd.
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
# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.utils import Registry, build_from_cfg

TRANSFORMER = Registry('Transformer')
LINEAR_LAYERS = Registry('linear layers')


def build_transformer(cfg, default_args=None):
    """Builder for Transformer."""
    return build_from_cfg(cfg, TRANSFORMER, default_args)


LINEAR_LAYERS.register_module('Linear', module=nn.Linear)


def build_linear_layer(cfg, *args, **kwargs):
    """Build linear layer.
    Args:
        cfg (None or dict): The linear layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate an linear layer.
        args (argument list): Arguments passed to the `__init__`
            method of the corresponding linear layer.
        kwargs (keyword arguments): Keyword arguments passed to the `__init__`
            method of the corresponding linear layer.
    Returns:
        nn.Module: Created linear layer.
    """
    if cfg is None:
        cfg_ = dict(type='Linear')
    else:
        if not isinstance(cfg, dict):
            raise TypeError('cfg must be a dict')
        if 'type' not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in LINEAR_LAYERS:
        raise KeyError(f'Unrecognized linear type {layer_type}')
    else:
        linear_layer = LINEAR_LAYERS.get(layer_type)

    layer = linear_layer(*args, **kwargs, **cfg_)

    return layer
