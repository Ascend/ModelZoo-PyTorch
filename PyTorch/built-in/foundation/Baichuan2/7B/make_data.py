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
