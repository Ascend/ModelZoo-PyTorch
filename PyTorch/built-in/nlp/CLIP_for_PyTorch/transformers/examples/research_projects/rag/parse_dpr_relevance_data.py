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
