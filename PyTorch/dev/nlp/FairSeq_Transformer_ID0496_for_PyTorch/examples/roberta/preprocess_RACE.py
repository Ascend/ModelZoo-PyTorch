#!/usr/bin/env python
#
# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
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
# ============================================================================
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import re


class InputExample:
    def __init__(self, paragraph, qa_list, label):
        self.paragraph = paragraph
        self.qa_list = qa_list
        self.label = label


def get_examples(data_dir, set_type):
    """
    Extract paragraph and question-answer list from each json file
    """
    examples = []

    levels = ["middle", "high"]
    set_type_c = set_type.split('-')
    if len(set_type_c) == 2:
        levels = [set_type_c[1]]
        set_type = set_type_c[0]
    for level in levels:
        cur_dir = os.path.join(data_dir, set_type, level)
        for filename in os.listdir(cur_dir):
            cur_path = os.path.join(cur_dir, filename)
            with open(cur_path, 'r') as f:
                cur_data = json.load(f)
                answers = cur_data["answers"]
                options = cur_data["options"]
                questions = cur_data["questions"]
                context = cur_data["article"].replace("\n", " ")
                context = re.sub(r'\s+', ' ', context)
                for i in range(len(answers)):
                    label = ord(answers[i]) - ord("A")
                    qa_list = []
                    question = questions[i]
                    for j in range(4):
                        option = options[i][j]
                        if "_" in question:
                            qa_cat = question.replace("_", option)
                        else:
                            qa_cat = " ".join([question, option])
                        qa_cat = re.sub(r'\s+', ' ', qa_cat)
                        qa_list.append(qa_cat)
                    examples.append(InputExample(context, qa_list, label))

    return examples


def main():
    """
    Helper script to extract paragraphs questions and answers from RACE datasets.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        help='input directory for downloaded RACE dataset',
    )
    parser.add_argument(
        "--output-dir",
        help='output directory for extracted data',
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    for set_type in ["train", "dev", "test-middle", "test-high"]:
        examples = get_examples(args.input_dir, set_type)
        qa_file_paths = [os.path.join(args.output_dir, set_type + ".input" + str(i + 1)) for i in range(4)]
        qa_files = [open(qa_file_path, 'w') for qa_file_path in qa_file_paths]
        outf_context_path = os.path.join(args.output_dir, set_type + ".input0")
        outf_label_path = os.path.join(args.output_dir, set_type + ".label")
        outf_context = open(outf_context_path, 'w')
        outf_label = open(outf_label_path, 'w')
        for example in examples:
            outf_context.write(example.paragraph + '\n')
            for i in range(4):
                qa_files[i].write(example.qa_list[i] + '\n')
            outf_label.write(str(example.label) + '\n')

        for f in qa_files:
            f.close()
        outf_label.close()
        outf_context.close()


if __name__ == '__main__':
    main()
