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

# 测试bart语言模型的预测效果
# bert4torch需要转换一下权重，见convert文件夹中

from transformers import BertTokenizer, BartForConditionalGeneration
tokenizer = BertTokenizer.from_pretrained("F:/Projects/pretrain_ckpt/bart/[FudanNLP_torch_base]/")
model = BartForConditionalGeneration.from_pretrained("F:/Projects/pretrain_ckpt/bart/[FudanNLP_torch_base]/")

input_ids = tokenizer.encode("北京是[MASK]的首都", return_tensors='pt')
pred_ids = model.generate(input_ids, num_beams=4, max_length=20)
print('transformers output: ', tokenizer.convert_ids_to_tokens(pred_ids[0]))
# 输出： ['[SEP]', '[CLS]', '北', '京', '是', '中', '国', '的', '首', '都', '[SEP]'] 



from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import Tokenizer
from bert4torch.snippets import AutoRegressiveDecoder
import torch

# bert配置
config_path = 'F:/Projects/pretrain_ckpt/bart/[FudanNLP_torch_base]/bert4torch_config.json'
checkpoint_path = 'F:/Projects/pretrain_ckpt/bart/[FudanNLP_torch_base]/bert4torch_pytorch_model.bin'
dict_path = 'F:/Projects/pretrain_ckpt/bart/[FudanNLP_torch_base]/vocab.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = Tokenizer(dict_path, do_lower_case=True)
model = build_transformer_model(config_path, checkpoint_path, model='bart', segment_vocab_size=0).to(device)

class AutoTitle(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps(default_rtype='logits')
    def predict(self, inputs, output_ids, states):
        return model.decoder.predict([output_ids] + inputs)[-1][:, -1, :]  # 保留最后一位

    def generate(self, text, topk=4):
        token_ids, _ = tokenizer.encode(text, maxlen=128)
        token_ids = torch.tensor([token_ids], device=device)
        encoder_output = model.encoder.predict([token_ids])
        output_ids = self.beam_search(encoder_output, topk=topk)  # 基于beam search
        return tokenizer.decode(output_ids.cpu().numpy())

autotitle = AutoTitle(start_id=102, end_id=tokenizer._token_end_id, maxlen=32, device=device)

print('bert4torch output: ', autotitle.generate("北京是[MASK]的首都"))