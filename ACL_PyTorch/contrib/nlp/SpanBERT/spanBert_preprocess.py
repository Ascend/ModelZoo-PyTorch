# Copyright 2022 Huawei Technologies Co., Ltd
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
import sys
sys.path.append(r"./SpanBERT/code")
import argparse
import collections
import json
import logging
import re
from io import open
import numpy as np
import torch
import os
from torch.utils.data import DataLoader, TensorDataset
from run_squad import *
from pytorch_pretrained_bert.tokenization import (BertTokenizer,
                                                  whitespace_tokenize)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)



def main(args):

    tokenizer = BertTokenizer.from_pretrained(
        args.model, do_lower_case=args.do_lower_case)
    eval_examples = read_squad_examples(
        input_file=args.dev_file, is_training=False,
        version_2_with_negative=args.version_2_with_negative)
    eval_features = convert_examples_to_features(
        examples=eval_examples,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length,
        is_training=False)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
    eval_dataloader = DataLoader(eval_data, batch_size=args.batch_size)
    input_ids_path = "./input_ids"
    input_mask_path = "./input_mask"
    segment_ids_path = "./segment_ids"
    if not os.path.exists(input_ids_path):
        os.makedirs(input_ids_path)
    if not os.path.exists(input_mask_path):
        os.makedirs(input_mask_path)
    if not os.path.exists(segment_ids_path):
        os.makedirs(segment_ids_path)
    i = -1
    for input_ids, input_mask, segment_ids, example_indices in eval_dataloader:
        i = i + 1
        print("[info] file", "===", i)
        
        input_ids_np = input_ids.numpy()
        input_mask_np = input_mask.numpy()
        segment_ids_np = segment_ids.numpy()
        input_ids_np.tofile(os.path.join(input_ids_path, str(i) + '.bin'))
        segment_ids_np.tofile(os.path.join(segment_ids_path, str(i) + '.bin'))
        input_mask_np.tofile(os.path.join(input_mask_path, str(i) + '.bin'))


if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", default='spanbert-base-cased', type=str)
        parser.add_argument("--dev_file", default='dev-v1.1.json', type=str,
                            help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
        parser.add_argument("--max_seq_length", default=512, type=int,
                            help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                                 "longer than this will be truncated, and sequences shorter than this will be padded.")
        parser.add_argument("--doc_stride", default=128, type=int,
                            help="When splitting up a long document into chunks, "
                                 "how much stride to take between chunks.")
        parser.add_argument("--max_query_length", default=64, type=int,
                            help="The maximum number of tokens for the question. Questions longer than this will "
                                 "be truncated to this length.")
        parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
        parser.add_argument('--version_2_with_negative', action='store_true',
                            help='If true, the SQuAD examples contain some that do not have an answer.')
        parser.add_argument("--batch_size", default=1, type=int,
                            help="When splitting up a long document into chunks, "
                                 "how much stride to take between chunks.")
        args = parser.parse_args()

        main(args)
