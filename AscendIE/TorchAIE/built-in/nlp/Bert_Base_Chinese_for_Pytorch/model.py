# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
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

import torch
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer


def build_tokenizer(tokenizer_name):
    tokenizer_kwargs = {
        'cache_dir': None,
        'use_fast': True,
        'revision': 'main',
        'use_auth_token': None
    }
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)
    return tokenizer


def build_base_model(tokenizer, model_path, config_path, device):
    config_kwargs = {
        'cache_dir': None,
        'revision': 'main',
        'use_auth_token': None
    }
    config = AutoConfig.from_pretrained(config_path, **config_kwargs)
    model = AutoModelForMaskedLM.from_pretrained(
        model_path,
        config=config,
        revision='main',
        use_auth_token=None
    )
    model.to(device=device)
    model.eval()
    model.resize_token_embeddings(len(tokenizer))
    return model


class RefineModel(torch.nn.Module):
    def __init__(self, tokenizer, model_path, config_path, device="cpu"):
        super(RefineModel, self).__init__()
        self._base_model = build_base_model(tokenizer, model_path, config_path, device)

    def forward(self, input_ids, attention_mask, token_type_ids):
        x = self._base_model(input_ids, attention_mask, token_type_ids)
        return x[0].argmax(dim=-1)
