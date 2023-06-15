# encoding=utf-8
# Copyright 2021 Huawei Technologies Co., Ltd
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

import os
import sys
import torch
from torch.utils.data import DataLoader,SequentialSampler, TensorDataset
sys.path.append(r"./albert_pytorch")
from albert_pytorch.model import tokenization_albert
from albert_pytorch.processors import glue_output_modes as output_modes
from albert_pytorch.processors import glue_processors as processors
from albert_pytorch.processors import glue_convert_examples_to_features as convert_examples_to_features
from albert_pytorch.processors import collate_fn
from albert_pytorch.tools.common import init_logger, logger
from albert_pytorch.model.modeling_albert import AlbertConfig, AlbertForSequenceClassification


def load_and_cache_examples(args, task, tokenizer, data_type='dev'):
    if args.local_rank not in [-1, 0]:  # and not evaluate:
        torch.distributed.barrier()
    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        data_type,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ['mnli', 'mnli-mm'] and 'roberta' in args.model_type:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]

        if data_type == 'train':
            examples = processor.get_train_examples(args.data_dir)
        elif data_type == 'dev':
            examples = processor.get_dev_examples(args.data_dir)
        else:
            examples = processor.get_test_examples(args.data_dir)

        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                label_list=label_list,
                                                max_seq_length=args.max_seq_length,
                                                output_mode=output_mode)
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0:  # and not evaluate:
        torch.distributed.barrier()
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels)
    return dataset


def load_data_model(ar):
    args = torch.load(ar.pth_arg_path)
    args.device = 'cpu'
    args.local_rank = -1
    args.batch_size = ar.batch_size
    args.max_seq_length = ar.max_seq_length

    prefix = ar.prefix_dir
    args.output_dir = ar.pth_dir
    args.data_dir = ar.data_dir
    args.vocab_file = prefix + args.vocab_file[1:]
    args.spm_model_file = prefix + args.spm_model_file[1:]

    logger.args = args
    tokenizer = tokenization_albert.FullTokenizer(vocab_file=args.vocab_file, do_lower_case=args.do_lower_case,
                                                  spm_model_file=args.spm_model_file)
    model = AlbertForSequenceClassification.from_pretrained(args.output_dir)
    test_dataset = load_and_cache_examples(args, args.task_name, tokenizer, data_type=ar.data_type)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size,
                                 drop_last=True,
                                 collate_fn=collate_fn)

    data, label = [], []
    for batch in test_dataloader:
        if len(batch[0]) != args.batch_size: 
            continue
        data.append({'input_ids': batch[0], 'attention_mask': batch[1], 'token_type_ids': batch[2]})
        label.append(batch[3])
    return tuple(data), model, label
