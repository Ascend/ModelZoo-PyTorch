# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import re
import os
import argparse
from tqdm import tqdm
from data_utils.tokenization_gpt2 import GPT2Tokenizer


def process_one_sent(sent, answers, cands, tokenizer, num_ids, split):
    pattern = re.compile(r"#idiom\d+#")
    res = pattern.findall(sent)
    start = 0
    L = []
    # Template(NOTE: A little different from the one in the paper): 
    # 上下文: P_predix <mask> P_postfix <eod> 选项0: I_0 <sep> 选项1: I_1 <sep> ... 选项9: I_9 <sep> <mask> 答案是: L
    # The P_prefix indicates the text before the idiom. The P_postfix inficates the text after the idiom. We insert <mask> between them for a placeholder of the idiom.
    # <mask> <eod> and <sep> are the special tokens in our vocabulary.
    while True:
        m = pattern.search(sent, start)
        if m is None:
            break
        
        cands_ids = []
        for i, cand in enumerate(cands):
            cands_ids.extend(tokenizer.encode("选项{}:".format(i)))
            cands_ids.extend(tokenizer.encode(cand.strip()))
            cands_ids.append(tokenizer.encoder["<sep>"])

        prefix = re.sub(pattern, "", sent[:m.start()])
        postfix = re.sub(pattern, "", sent[m.end():])

        context_ids = tokenizer.encode("上下文:") + tokenizer.encode(prefix.strip()) + [tokenizer.encoder["<mask>"]] + tokenizer.encode(postfix.strip()) + [tokenizer.encoder["<eod>"]]

        ques_ids = [tokenizer.encoder["<mask>"]] + tokenizer.encode("答案是:") + [num_ids[answers[m.group()]]]
        
        ids = context_ids + cands_ids + ques_ids

        L.append({
            "sent": ids,
            "truth": answers[m.group()],
        })

        start = m.end()
    
    return L


def preprocess(data, tokenizer, split):

    # Get the token id of "0", "1", "2", ... "9"
    num_ids = [tokenizer.encode("选项{}:".format(i))[1] for i in range(10)]

    lines, ans_d = data
    
    all_data = []
    for line in tqdm(lines, desc="Preprocessing {}".format(split)):
        jobj = json.loads(line)
        for sent in jobj["content"]:
            samples = process_one_sent(sent, ans_d, jobj["candidates"], tokenizer, num_ids, split)
            all_data.extend(samples)

    return num_ids, all_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_dir", default=None, type=str, help="The input dir of original ChID data.")
    parser.add_argument("--tokenizer_path", type=str, help="The tokenizer path.", default="./bpe_3w_new")
    parser.add_argument("--output_dir", type=str, help="The processed data output dir.")

    args = parser.parse_args()

    tokenizer = GPT2Tokenizer(os.path.join(args.tokenizer_path, 'vocab.json'), os.path.join(args.tokenizer_path, 'chinese_vocab.model'))

    os.makedirs(args.output_dir, exist_ok=True)

    for split in ["train", "dev", "test"]:
        with open(os.path.join(args.data_dir, "{}.json".format(split)), "r") as f:
            lines = f.readlines()

        with open(os.path.join(args.data_dir, "{}_answer.json".format(split)), "r") as f:
            ans_d = json.load(f)

        num_ids, all_data = preprocess((lines, ans_d), tokenizer, split)

        with open(os.path.join(args.output_dir, "{}.json".format(split)), "w") as f:
            json.dump((num_ids, all_data), f, indent=4, ensure_ascii=False)
