# Copyright 2021 Huawei Technologies Co., Ltd
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


from mmcv.utils import Registry, build_from_cfg

BBOX_ASSIGNERS = Registry('bbox_assigner')
BBOX_SAMPLERS = Registry('bbox_sampler')
BBOX_CODERS = Registry('bbox_coder')


def build_assigner(cfg, **default_args):
    """Builder of box assigner."""
    return build_from_cfg(cfg, BBOX_ASSIGNERS, default_args)


def build_sampler(cfg, **default_args):
    """Builder of box sampler."""
    return build_from_cfg(cfg, BBOX_SAMPLERS, default_args)


def build_bbox_coder(cfg, **default_args):
    """Builder of box coder."""
    return build_from_cfg(cfg, BBOX_CODERS, default_args)
