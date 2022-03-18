# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
from torch import nn

from alphapose.utils import Registry, build_from_cfg, retrieve_from_cfg


SPPE = Registry('sppe')
LOSS = Registry('loss')
DATASET = Registry('dataset')


def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_sppe(cfg, preset_cfg, **kwargs):
    default_args = {
        'PRESET': preset_cfg,
    }
    for key, value in kwargs.items():
        default_args[key] = value
    return build(cfg, SPPE, default_args=default_args)


def build_loss(cfg):
    return build(cfg, LOSS)


def build_dataset(cfg, preset_cfg, **kwargs):
    exec(f'from ..datasets import {cfg.TYPE}')
    default_args = {
        'PRESET': preset_cfg,
    }
    for key, value in kwargs.items():
        default_args[key] = value
    return build(cfg, DATASET, default_args=default_args)


def retrieve_dataset(cfg):
    exec(f'from ..datasets import {cfg.TYPE}')
    return retrieve_from_cfg(cfg, DATASET)
