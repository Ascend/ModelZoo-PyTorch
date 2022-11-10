# -*- coding: utf-8 -*-
# BSD 3-Clause License
#
# Copyright (c) 2017
# All rights reserved.
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ==========================================================================

from transformers import XLNetTokenizer, XLNetModel
import torch

pretrained_model = "F:/Projects/pretrain_ckpt/xlnet/[hit_torch_base]--chinese-xlnet-base"
tokenizer = XLNetTokenizer.from_pretrained(pretrained_model)
model = XLNetModel.from_pretrained(pretrained_model)

inputs = tokenizer(["你好啊，我叫张三", "天气不错啊"], padding=True, return_tensors="pt")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
print('--------transformers last_hidden_state--------\n', last_hidden_states)

# ----------------------bert4torch配置----------------------
from bert4torch.models import build_transformer_model
config_path = f'{pretrained_model}/bert4torch_config.json'
checkpoint_path = f'{pretrained_model}/pytorch_model.bin'

model = build_transformer_model(
    config_path,
    checkpoint_path=checkpoint_path,
    model='xlnet',
    # with_lm=True
    token_pad_ids=tokenizer.pad_token_id,
)

print('--------bert4torch last_hidden_state--------\n', model.predict([inputs['input_ids'], inputs['token_type_ids']]))