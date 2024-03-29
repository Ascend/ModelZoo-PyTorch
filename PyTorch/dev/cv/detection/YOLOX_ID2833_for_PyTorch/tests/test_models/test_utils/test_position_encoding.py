
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
import pytest
import torch

from mmdet.models.utils import (LearnedPositionalEncoding,
                                SinePositionalEncoding)


def test_sine_positional_encoding(num_feats=16, batch_size=2):
    # test invalid type of scale
    with pytest.raises(AssertionError):
        module = SinePositionalEncoding(
            num_feats, scale=(3., ), normalize=True)

    module = SinePositionalEncoding(num_feats)
    h, w = 10, 6
    mask = (torch.rand(batch_size, h, w) > 0.5).to(torch.int)
    assert not module.normalize
    out = module(mask)
    assert out.shape == (batch_size, num_feats * 2, h, w)

    # set normalize
    module = SinePositionalEncoding(num_feats, normalize=True)
    assert module.normalize
    out = module(mask)
    assert out.shape == (batch_size, num_feats * 2, h, w)


def test_learned_positional_encoding(num_feats=16,
                                     row_num_embed=10,
                                     col_num_embed=10,
                                     batch_size=2):
    module = LearnedPositionalEncoding(num_feats, row_num_embed, col_num_embed)
    assert module.row_embed.weight.shape == (row_num_embed, num_feats)
    assert module.col_embed.weight.shape == (col_num_embed, num_feats)
    h, w = 10, 6
    mask = torch.rand(batch_size, h, w) > 0.5
    out = module(mask)
    assert out.shape == (batch_size, num_feats * 2, h, w)
