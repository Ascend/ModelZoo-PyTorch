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
"""
This script reads DPR retriever training data and parses each datapoint. We save a line per datapoint.
Each line consists of the query followed by a tab-separated list of Wikipedia page titles constituting
positive contexts for a given query.
"""

import argparse
import json

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--src_path",
        type=str,
        default="biencoder-nq-dev.json",
        help="Path to raw DPR training data",
    )
    parser.add_argument(
        "--evaluation_set",
        type=str,
        help="where to store parsed evaluation_set file",
    )
    parser.add_argument(
        "--gold_data_path",
        type=str,
        help="where to store parsed gold_data_path file",
    )
    args = parser.parse_args()

    with open(args.src_path, "r") as src_file, open(args.evaluation_set, "w") as eval_file, open(
        args.gold_data_path, "w"
    ) as gold_file:
        dpr_records = json.load(src_file)
        for dpr_record in tqdm(dpr_records):
            question = dpr_record["question"]
            contexts = [context["title"] for context in dpr_record["positive_ctxs"]]
            eval_file.write(question + "\n")
            gold_file.write("\t".join(contexts) + "\n")


if __name__ == "__main__":
    main()
