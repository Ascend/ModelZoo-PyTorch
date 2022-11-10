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

#! -*- coding: utf-8 -*-
# 调用预训练的t5-chinese模型直接做预测,使用的BertTokenizer
# t5使用的是t5.1.0的结构

import torch
from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import Tokenizer, load_vocab
from bert4torch.snippets import AutoRegressiveDecoder

# bert配置
config_path = 'F:/Projects/pretrain_ckpt/t5/[uer_t5_torch_small]--t5-small-chinese-cluecorpussmall/bert4torch_config.json'
checkpoint_path = 'F:/Projects/pretrain_ckpt/t5/[uer_t5_torch_small]--t5-small-chinese-cluecorpussmall/pytorch_model.bin'
dict_path = 'F:/Projects/pretrain_ckpt/t5/[uer_t5_torch_small]--t5-small-chinese-cluecorpussmall/vocab.txt'

# config_path = 'F:/Projects/pretrain_ckpt/t5/[uer_t5_torch_base]--t5-base-chinese-cluecorpussmall/bert4torch_config.json'
# checkpoint_path = 'F:/Projects/pretrain_ckpt/t5/[uer_t5_torch_base]--t5-base-chinese-cluecorpussmall/pytorch_model.bin'
# dict_path = 'F:/Projects/pretrain_ckpt/t5/[uer_t5_torch_base]--t5-base-chinese-cluecorpussmall/vocab.txt'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 加载并精简词表，建立分词器
token_dict = load_vocab(
    dict_path=dict_path,
    simplified=False,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)

model = build_transformer_model(
    config_path,
    checkpoint_path,
    model='t5.1.0',
    segment_vocab_size=0
).to(device)

class AutoTitle(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps(default_rtype='logits')
    def predict(self, inputs, output_ids, states):
        token_ids = inputs[0] 
        return model.predict([[token_ids], [output_ids]])[-1][:, -1, :]  # 保留最后一位

    def generate(self, text, topk=1, topp=0.95):
        token_ids, _ = tokenizer.encode(text, maxlen=256)
        output_ids = self.beam_search([token_ids], topk=topk)  # 基于beam search
        return tokenizer.decode(output_ids.cpu().numpy())

autotitle = AutoTitle(start_id=tokenizer._token_start_id, end_id=1, maxlen=32, device=device)  # 这里end_id可以设置为tokenizer._token_end_id这样结果更短

if __name__ == '__main__':
    print(autotitle.generate('中国的首都是extra0京'))

