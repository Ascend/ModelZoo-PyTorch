# Copyright 2023 Huawei Technologies Co., Ltd
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

import pytest

import torch
from open_clip.hf_model import _POOLERS, HFTextEncoder
from transformers import AutoConfig
from transformers.modeling_outputs import BaseModelOutput
# test poolers
def test_poolers():
    bs, sl, d = 2, 10, 5
    h = torch.arange(sl).repeat(bs).reshape(bs, sl)[..., None] * torch.linspace(0.2, 1., d)
    mask = torch.ones(bs, sl, dtype=torch.long)
    mask[:2, 6:] = 0
    x = BaseModelOutput(h)
    for name, cls in _POOLERS.items():
        pooler = cls()
        res = pooler(x, mask)
        assert res.shape == (bs, d), f"{name} returned wrong shape"

# test HFTextEncoder
@pytest.mark.parametrize("model_id", ["arampacha/roberta-tiny", "roberta-base", "xlm-roberta-base", "google/mt5-base"])
def test_pretrained_text_encoder(model_id):
    bs, sl, d = 2, 10, 64
    cfg = AutoConfig.from_pretrained(model_id)
    model = HFTextEncoder(model_id, d, proj='linear')
    x = torch.randint(0, cfg.vocab_size, (bs, sl))
    with torch.no_grad():
        emb = model(x)

    assert emb.shape == (bs, d)
