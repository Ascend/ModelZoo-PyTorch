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
# 基本测试：中文GPT模型，base版本，华为开源的
# 权重链接: https://pan.baidu.com/s/1-FB0yl1uxYDCGIRvU1XNzQ 提取码: xynn，这里使用的是转pytorch后的模型文件
# 参考项目：https://github.com/bojone/chinese-gen

import torch
from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import Tokenizer
from bert4torch.snippets import AutoRegressiveDecoder

config_path = 'F:/Projects/pretrain_ckpt/bert/[huawei_noah_tf_base]--chinese_nezha_gpt_L-12_H-768_A-12/config.json'
checkpoint_path = 'F:/Projects/pretrain_ckpt/bert/[huawei_noah_tf_base]--chinese_nezha_gpt_L-12_H-768_A-12/pytorch_model.bin'
dict_path = 'F:/Projects/pretrain_ckpt/bert/[huawei_noah_tf_base]--chinese_nezha_gpt_L-12_H-768_A-12/vocab.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器

model = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    segment_vocab_size=0,  # 去掉segmeng_ids输入
    application='lm',
).to(device)  # 建立模型，加载权重


class ArticleCompletion(AutoRegressiveDecoder):
    """基于随机采样的文章续写
    """
    @AutoRegressiveDecoder.wraps(default_rtype='logits')
    def predict(self, inputs, output_ids, states):
        token_ids = torch.cat([inputs[0], output_ids], 1)
        _, mlm_scores = model.predict([token_ids])
        return mlm_scores[:, -1, :]

    def generate(self, text, n=1, topp=0.95):
        token_ids = tokenizer.encode(text)[0][:-1]
        results = self.random_sample([token_ids], n, topp=topp)  # 基于随机采样
        return [text + tokenizer.decode(ids.cpu().numpy()) for ids in results]


article_completion = ArticleCompletion(
    start_id=None,
    end_id=511,  # 511是中文句号
    maxlen=256,
    minlen=128,
    device=device
)

print(article_completion.generate(u'今天天气不错'))
"""
部分结果：
>>> article_completion.generate(u'今天天气不错')
[u'今天天气不错。昨天的天气是多云到晴的天气，今天的天气还不错，不会太冷。明后两天天气还是比较好的。不过今天的天气比较闷热，最高温度在30℃左右，明后两天天气会更加热。预计今天的最高温度为30℃，明后两天的最   高温度为32℃左右，今天的最高气温将在30℃左右。（记者李莉）。新华网重庆频道诚邀广大网友投稿，您可以用相机或手机记录下身边的感人故事，精彩瞬间。请将作者、拍摄时间、地点和简要说明连同照片发给我们，我们将精选其中的好图、美图在页面上展示，让所有新华网友共赏。[投稿] 。本报讯(记者陈敏华) 今年上半年，重庆市各级公安机关在全力抓好']

>>> article_completion.generate(u'双十一')
[u'双十一大是中国共产党在新的历史起点上召开的一次十分重要的代表大会, 是全面落实科学发展观、推进中国特色社会主义伟大事业的一次重要会议。会议的召开, 是党和政府对新世纪新阶段我国改革开放和社会主义现代化建设 事业的新的历史任务的一次重要总动员, 必将对我们党全面推进党的建']

>>> article_completion.generate(u'科学空间')
[u'科学空间站上的两个机器人在进入轨道后，一边在轨道上工作，一边用它们的身体和心脏在空间站上的一个大气层进行活动，以确保它们在进入地球之后不会因太阳风暴而受到影响；而另外一个机器人则在进入轨道的过程中，通 过机器人与地球上的大气层相互作用，使地球的大气层不断地向地球的大气层中转移，以使其能够在空间站上工作，并且使用它们的身体和心脏来完成它们的各种任务。']
"""
