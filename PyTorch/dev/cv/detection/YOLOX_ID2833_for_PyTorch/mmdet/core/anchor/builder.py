
# Copyright 2022 Huawei Technologies Co., Ltd
#
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

# Copyright (c) Open-MMLab. All rights reserved.    
# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from mmcv.utils import Registry, build_from_cfg

PRIOR_GENERATORS = Registry('Generator for anchors and points')

ANCHOR_GENERATORS = PRIOR_GENERATORS


def build_prior_generator(cfg, default_args=None):
    return build_from_cfg(cfg, PRIOR_GENERATORS, default_args)


def build_anchor_generator(cfg, default_args=None):
    warnings.warn(
        '``build_anchor_generator`` would be deprecated soon, please use '
        '``build_prior_generator`` ')
    return build_prior_generator(cfg, default_args=default_args)
