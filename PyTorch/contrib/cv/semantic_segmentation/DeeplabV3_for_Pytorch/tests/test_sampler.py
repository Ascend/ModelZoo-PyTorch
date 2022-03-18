#     Copyright 2021 Huawei
#     Copyright 2021 Huawei Technologies Co., Ltd
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
#

import pytest
import torch

from mmseg.core import OHEMPixelSampler
from mmseg.models.decode_heads import FCNHead


def _context_for_ohem():
    return FCNHead(in_channels=32, channels=16, num_classes=19)


def test_ohem_sampler():

    with pytest.raises(AssertionError):
        # seg_logit and seg_label must be of the same size
        sampler = OHEMPixelSampler(context=_context_for_ohem())
        seg_logit = torch.randn(1, 19, 45, 45)
        seg_label = torch.randint(0, 19, size=(1, 1, 89, 89))
        sampler.sample(seg_logit, seg_label)

    # test with thresh
    sampler = OHEMPixelSampler(
        context=_context_for_ohem(), thresh=0.7, min_kept=200)
    seg_logit = torch.randn(1, 19, 45, 45)
    seg_label = torch.randint(0, 19, size=(1, 1, 45, 45))
    seg_weight = sampler.sample(seg_logit, seg_label)
    assert seg_weight.shape[0] == seg_logit.shape[0]
    assert seg_weight.shape[1:] == seg_logit.shape[2:]
    assert seg_weight.sum() > 200

    # test w.o thresh
    sampler = OHEMPixelSampler(context=_context_for_ohem(), min_kept=200)
    seg_logit = torch.randn(1, 19, 45, 45)
    seg_label = torch.randint(0, 19, size=(1, 1, 45, 45))
    seg_weight = sampler.sample(seg_logit, seg_label)
    assert seg_weight.shape[0] == seg_logit.shape[0]
    assert seg_weight.shape[1:] == seg_logit.shape[2:]
    assert seg_weight.sum() == 200
