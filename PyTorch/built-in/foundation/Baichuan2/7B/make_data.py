# coding=utf-8
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

# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# Need to call this before importing transformers.

import json
import os
import random

import pandas as pd

data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

# 注意将csv文件的路径替换为实际的路径
pd_all = pd.read_csv('path_to_waimai_10k.csv')

datas = []
for row in pd_all.itertuples():
    datas.append(json.dumps({"review": row.review,
                             "label": "积极" if row.label else "消极"}, ensure_ascii=False) + '\n')
random.shuffle(datas)

if len(datas) > 10000:
    split_idx = 10000
else:
    split_idx = int(0.8 * len(datas))
with open(os.path.join(data_dir, "train.jsonl"), 'w', encoding='utf-8') as f:
    f.writelines(datas[:split_idx])
with open(os.path.join(data_dir, "eval.jsonl"), 'w', encoding='utf-8') as f:
    f.writelines(datas[split_idx:])
