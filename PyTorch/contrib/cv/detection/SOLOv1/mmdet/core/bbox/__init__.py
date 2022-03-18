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

from .assigners import AssignResult, BaseAssigner, MaxIoUAssigner
from .bbox_target import bbox_target
from .geometry import bbox_overlaps
from .samplers import (BaseSampler, CombinedSampler,
                       InstanceBalancedPosSampler, IoUBalancedNegSampler,
                       PseudoSampler, RandomSampler, SamplingResult)
from .transforms import (bbox2delta, bbox2result, bbox2roi, bbox_flip,
                         bbox_mapping, bbox_mapping_back, delta2bbox,
                         distance2bbox, roi2bbox)

from .assign_sampling import (  # isort:skip, avoid recursive imports
    assign_and_sample, build_assigner, build_sampler)

__all__ = [
    'bbox_overlaps', 'BaseAssigner', 'MaxIoUAssigner', 'AssignResult',
    'BaseSampler', 'PseudoSampler', 'RandomSampler',
    'InstanceBalancedPosSampler', 'IoUBalancedNegSampler', 'CombinedSampler',
    'SamplingResult', 'build_assigner', 'build_sampler', 'assign_and_sample',
    'bbox2delta', 'delta2bbox', 'bbox_flip', 'bbox_mapping',
    'bbox_mapping_back', 'bbox2roi', 'roi2bbox', 'bbox2result',
    'distance2bbox', 'bbox_target'
]
