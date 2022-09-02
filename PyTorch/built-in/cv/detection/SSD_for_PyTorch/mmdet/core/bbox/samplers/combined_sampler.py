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
from ..builder import BBOX_SAMPLERS, build_sampler
from .base_sampler import BaseSampler


@BBOX_SAMPLERS.register_module()
class CombinedSampler(BaseSampler):
    """A sampler that combines positive sampler and negative sampler."""

    def __init__(self, pos_sampler, neg_sampler, **kwargs):
        super(CombinedSampler, self).__init__(**kwargs)
        self.pos_sampler = build_sampler(pos_sampler, **kwargs)
        self.neg_sampler = build_sampler(neg_sampler, **kwargs)

    def _sample_pos(self, **kwargs):
        """Sample positive samples."""
        raise NotImplementedError

    def _sample_neg(self, **kwargs):
        """Sample negative samples."""
        raise NotImplementedError
