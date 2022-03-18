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


def process_one_sent_eval(tokenizer, sent, answers, candidates):
    pattern = re.compile(r"#idiom\d+#")
    res = pattern.findall(sent)
    start = 0
    L = []
    # fill the candidate idioms into the sentence to create candidate passages
    # NOTE: there may exist more than one blank in a sentence
    while True:
        m = pattern.search(sent, start)
        if m is None:
            break
        L.append({
            "cands": [],
            "truth": answers[m.group()]
        })
        for idm in candidates:
            cand = sent[:m.start()] + idm + sent[m.end():]
            # replace other blanks by ""
            cand = re.sub(pattern, "", cand)
            ids = tokenizer.encode(cand)
            L[-1]["cands"].append(ids)
        start = m.end()
    
    return L


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_dir", default=None, type=str, help="The input dir of original ChID data.")
    parser.add_argument("--tokenizer_path", type=str, help="The tokenizer path.", default="./bpe_3w_new")
    parser.add_argument("--output_dir", type=str, help="The processed data output dir.")

    args = parser.parse_args()

    tokenizer = GPT2Tokenizer(os.path.join(args.tokenizer_path, 'vocab.json'), os.path.join(args.tokenizer_path, 'chinese_vocab.model'))

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.data_dir, "idiomDict.json"), "r") as f:
        idiom_d = json.load(f)
    
    idioms = list(idiom_d.keys())

    for split in ["test"]:
        # for zero-shot setting, we only consider test set
        with open(os.path.join(args.data_dir, "{}.json".format(split)), "r") as f:
            lines = f.readlines()

        with open(os.path.join(args.data_dir, "{}_answer.json".format(split)), "r") as f:
            ans_d = json.load(f)

        all_data = {
            "contents": [],
            "sids": [],
            "labels": [],
            "cids": []
        }
        sid = 0
        for line in tqdm(lines, desc="Preprocessing {}".format(split)):
            jobj = json.loads(line)
            for sent in jobj["content"]:
                sample_L = process_one_sent_eval(tokenizer, sent, ans_d, jobj["candidates"])
                for samp in sample_L:
                    all_data["contents"].extend(samp["cands"])
                    all_data["sids"].extend([sid for _ in samp["cands"]])
                    all_data["cids"].extend([i for i in range(len(samp["cands"]))])
                    all_data["labels"].append(samp["truth"])
                    sid += 1

        with open(os.path.join(args.output_dir, "{}.json".format(split)), "w") as f:
            json.dump(all_data, f, indent=4, ensure_ascii=False)

        print(len(all_data["contents"]))

    with open(os.path.join(args.output_dir, "idioms.json"), "w") as f:
        json.dump(idioms, f, indent=4, ensure_ascii=False)
