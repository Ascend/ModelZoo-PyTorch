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
from __future__ import absolute_import, division, print_function
import sys
sys.path.append(r"./SpanBERT/code")
import argparse
import collections
import json
import logging
import math
import os
import random
import time
import re
import string
from io import open

import numpy as np
import torch
from run_squad import *
from torch.utils.data import DataLoader, TensorDataset
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForQuestionAnswering
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.tokenization import (BasicTokenizer,
                                                  BertTokenizer,
                                                  whitespace_tokenize)

import pickle
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate(args, eval_dataset, eval_dataloader,
             eval_examples, eval_features, na_prob_thresh=1.0, pred_only=False):
    all_results = []
    filepath = args.bin_dir
    all_file = []
    files = os.listdir(filepath)
    for fi in files:
        fi_d = os.path.join(filepath,fi)            
        if os.path.isdir(fi_d):
            all_file.append(fi_d)
    bin_path = all_file[0]

    for idx, (input_ids, input_mask, segment_ids, example_indices) in enumerate(eval_dataloader):
        batch_start_logits = np.fromfile('{}/{}_0.bin'.format(bin_path, idx), dtype='float32')
        batch_end_logits = np.fromfile('{}/{}_1.bin'.format(bin_path, idx), dtype='float32')
        batch_start_logits = torch.from_numpy(batch_start_logits) 
        batch_end_logits = torch.from_numpy(batch_end_logits)
        batch_start_logits = torch.reshape(batch_start_logits, (-1, 512))
        batch_end_logits = torch.reshape(batch_end_logits, (-1, 512))
        for i, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            all_results.append(RawResult(unique_id=unique_id,
                                         start_logits=start_logits,
                                         end_logits=end_logits))
    preds, nbest_preds, na_probs = \
        make_predictions(eval_examples, eval_features, all_results,
                         args.n_best_size, args.max_answer_length,
                         args.do_lower_case, args.verbose_logging,
                         args.version_2_with_negative)

    if pred_only:
        if args.version_2_with_negative:
            for k in preds:
                if na_probs[k] > na_prob_thresh:
                    preds[k] = ''
        return {}, preds, nbest_preds

    if args.version_2_with_negative:
        qid_to_has_ans = make_qid_to_has_ans(eval_dataset)
        has_ans_qids = [k for k, v in qid_to_has_ans.items() if v]
        no_ans_qids = [k for k, v in qid_to_has_ans.items() if not v]
        exact_raw, f1_raw = get_raw_scores(eval_dataset, preds)
        exact_thresh = apply_no_ans_threshold(exact_raw, na_probs, qid_to_has_ans, na_prob_thresh)
        f1_thresh = apply_no_ans_threshold(f1_raw, na_probs, qid_to_has_ans, na_prob_thresh)
        result = make_eval_dict(exact_thresh, f1_thresh)
        if has_ans_qids:
            has_ans_eval = make_eval_dict(exact_thresh, f1_thresh, qid_list=has_ans_qids)
            merge_eval(result, has_ans_eval, 'HasAns')
        if no_ans_qids:
            no_ans_eval = make_eval_dict(exact_thresh, f1_thresh, qid_list=no_ans_qids)
            merge_eval(result, no_ans_eval, 'NoAns')
        find_all_best_thresh(result, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans)
        for k in preds:
            if na_probs[k] > result['best_f1_thresh']:
                preds[k] = ''
    else:
        exact_raw, f1_raw = get_raw_scores(eval_dataset, preds)
        result = make_eval_dict(exact_raw, f1_raw)
    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
    return result, preds, nbest_preds


def main(args):
    n_gpu = 1 # torch.cuda.device_count()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = BertTokenizer.from_pretrained(
        args.model, do_lower_case=args.do_lower_case)

    if not args.eval_test:
        with open(args.dev_file) as f:
            dataset_json = json.load(f)
        eval_dataset = dataset_json['data']
        eval_examples = read_squad_examples(
            input_file=args.dev_file, is_training=False,
            version_2_with_negative=args.version_2_with_negative)
        start_time = time.time()
        if os.path.exists(args.data_file):
          with open(args.data_file, "rb") as handle:
              eval_features = pickle.load(handle)
        else:
            eval_features = convert_examples_to_features(
                examples=eval_examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length,
                is_training=False)
            with open(args.data_file, "wb") as handle:
              pickle.dump(eval_features, handle)
        print("after load eval_feartures: ", time.time() - start_time) # 109.45532488822937
        logger.info("***** Dev *****")
        logger.info("  Num orig examples = %d", len(eval_examples))
        logger.info("  Num split examples = %d", len(eval_features))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
        eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size)
    

    if args.do_eval:
        na_prob_thresh = 1.0
        if args.version_2_with_negative:
            eval_result_file = os.path.join(args.output_dir, "eval_results.txt")
            if os.path.isfile(eval_result_file):
                with open(eval_result_file) as f:
                    for line in f.readlines():
                        if line.startswith('best_f1_thresh'):
                            na_prob_thresh = float(line.strip().split()[-1])
                            logger.info("na_prob_thresh = %.6f" % na_prob_thresh)

        result, preds, _ = \
            evaluate(args, eval_dataset,
                     eval_dataloader, eval_examples, eval_features,
                     na_prob_thresh=na_prob_thresh,
                     pred_only=args.eval_test)
        with open(os.path.join(args.output_dir, "predictions.json"), "w") as writer:
            writer.write(json.dumps(preds, indent=4) + "\n")


if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", default=None, type=str, required=True)
        parser.add_argument("--output_dir", default='./', type=str,
                            help="The output directory where the model checkpoints and predictions will be written.")
        parser.add_argument("--bin_dir", default=None, type=str, required=True,
                            help="The output directory where the model checkpoints and predictions will be written.")
        parser.add_argument("--dev_file", default=None, type=str,
                            help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
        parser.add_argument("--max_seq_length", default=384, type=int,
                            help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                                 "longer than this will be truncated, and sequences shorter than this will be padded.")
        parser.add_argument("--doc_stride", default=128, type=int,
                            help="When splitting up a long document into chunks, "
                                 "how much stride to take between chunks.")
        parser.add_argument("--max_query_length", default=64, type=int,
                            help="The maximum number of tokens for the question. Questions longer than this will "
                                 "be truncated to this length.")
        parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
        parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
        parser.add_argument("--eval_test", action='store_true', help='Wehther to run eval on the test set.')
        parser.add_argument("--eval_batch_size", default=1, type=int, help="Total batch size for predictions.")
        parser.add_argument("--eval_metric", default='f1', type=str)
        parser.add_argument("--n_best_size", default=20, type=int,
                            help="The total number of n-best predictions to generate in the nbest_predictions.json "
                                 "output file.")
        parser.add_argument("--max_answer_length", default=30, type=int,
                            help="The maximum length of an answer that can be generated. "
                                 "This is needed because the start "
                                 "and end predictions are not conditioned on one another.")
        parser.add_argument("--verbose_logging", action='store_true',
                            help="If true, all of the warnings related to data processing will be printed. "
                                 "A number of warnings are expected for a normal SQuAD evaluation.")
        parser.add_argument("--no_cuda", action='store_true',
                            help="Whether not to use CUDA when available")
        parser.add_argument('--seed', type=int, default=42,
                            help="random seed for initialization")
        parser.add_argument('--fp16', action='store_true',
                            help="Whether to use 16-bit float precision instead of 32-bit")
        parser.add_argument('--version_2_with_negative', action='store_true',
                            help='If true, the SQuAD examples contain some that do not have an answer.')

        # 序列化及反序列化
        parser.add_argument("--data_file", default="data.pkl"
                            , type=str, help="filename to save data")
        args = parser.parse_args()

        main(args)
