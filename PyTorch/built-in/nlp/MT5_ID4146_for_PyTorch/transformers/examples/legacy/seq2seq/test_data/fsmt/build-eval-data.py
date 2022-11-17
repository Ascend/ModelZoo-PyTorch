# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#!/usr/bin/env python

import io
import json
import subprocess


pairs = [
    ["en", "ru"],
    ["ru", "en"],
    ["en", "de"],
    ["de", "en"],
]

n_objs = 8


def get_all_data(pairs, n_objs):
    text = {}
    for src, tgt in pairs:
        pair = f"{src}-{tgt}"
        cmd = f"sacrebleu -t wmt19 -l {pair} --echo src".split()
        src_lines = subprocess.run(cmd, stdout=subprocess.PIPE).stdout.decode("utf-8").splitlines()
        cmd = f"sacrebleu -t wmt19 -l {pair} --echo ref".split()
        tgt_lines = subprocess.run(cmd, stdout=subprocess.PIPE).stdout.decode("utf-8").splitlines()
        text[pair] = {"src": src_lines[:n_objs], "tgt": tgt_lines[:n_objs]}
    return text


text = get_all_data(pairs, n_objs)
filename = "./fsmt_val_data.json"
with io.open(filename, "w", encoding="utf-8") as f:
    bleu_data = json.dump(text, f, indent=2, ensure_ascii=False)
