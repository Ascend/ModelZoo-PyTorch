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

"""
Usage:
python3 -m fastchat.data.inspect --in sharegpt_20230322_clean_lang_split.json
"""
import argparse
import json

import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, required=True)
    parser.add_argument("--begin", type=int)
    args = parser.parse_args()

    content = json.load(open(args.in_file, "r"))
    for sample in tqdm.tqdm(content[args.begin :]):
        print(f"id: {sample['id']}")
        for conv in sample["conversations"]:
            print(conv["from"] + ": ")
            print(conv["value"])
            input()
