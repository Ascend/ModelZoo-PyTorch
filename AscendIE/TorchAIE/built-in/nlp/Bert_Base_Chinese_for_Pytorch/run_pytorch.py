# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
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
import os
import argparse
import time
import torch
from torch.utils.data import SequentialSampler, DataLoader

import numpy as np
from tqdm import tqdm
from datasets import load_metric, load_dataset
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

from transformers import BertTokenizerFast, DataCollatorForLanguageModeling, set_seed

import torch_aie
from torch_aie import _enums


def build_tokenizer(tokenizer_name):
    tokenizer_kwargs = {
        'cache_dir': None,
        'use_fast': True,
        'revision': 'main',
        'use_auth_token': None
    }
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)
    return tokenizer

def build_base_model(tokenizer, model_path, config_path, device):
    config_kwargs = {
        'cache_dir': None,
        'revision': 'main',
        'use_auth_token': None
    }
    config = AutoConfig.from_pretrained(config_path, **config_kwargs)
    model = AutoModelForMaskedLM.from_pretrained(
        model_path,
        config=config,
        revision='main',
        use_auth_token=None
    )
    model.to(device=device)
    model.eval()
    model.resize_token_embeddings(len(tokenizer))
    return model

class RefineModel(torch.nn.Module):
    def __init__(self, tokenizer, model_path, config_path, device="cpu"):
        super(RefineModel, self).__init__()
        self._base_model = build_base_model(tokenizer, model_path, config_path, device)

    def forward(self, input_ids, attention_mask, token_type_ids):
        x = self._base_model(input_ids, attention_mask, token_type_ids)
        return x[0]

def tokenize_process(dataset, tokenizer_name, max_seq_length):
    tokenizer_kwargs = {
        'cache_dir': None,
        'use_feat': None,
        'revision': 'main',
        'use_auth_token': None
    }
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)
    max_seq_length = min(max_seq_length, tokenizer.model_max_length)

    def tokenizer_function(examples):
        # remove empty lines
        examples['text'] = [
            line for line in examples['text'] if len(line) > 0 and not line.isspace()
        ]
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=max_seq_length,
            return_special_tokens_mask=True
        )

    tokenized_datasets = dataset.map(
        tokenizer_function,
        batched=True,
        remove_columns=['text'],
        load_from_cache_file=False,
        desc="Running tokenizer on dataset line by line"
    )
    return tokenized_datasets['validation'], tokenizer


def build_dataloader(dataset, tokenizer, batch_size):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=0.15
    )
    eval_sampler = SequentialSampler(dataset)
    return DataLoader(
        dataset,
        sampler=eval_sampler,
        batch_size=batch_size,
        collate_fn=data_collator,
        drop_last=True
    )


def preprocess(data_files, model_dir, save_dir, batch_size, max_seq_length):
    dataset = load_dataset("text", data_files={"validation": data_files})
    dataset, tokenizer = tokenize_process(dataset,
                                          tokenizer_name=model_dir,
                                          max_seq_length=max_seq_length)
    dataloader = build_dataloader(dataset, tokenizer, batch_size)


    # build tokenizer
    tokenizer = build_tokenizer(tokenizer_name=model_dir)
    # build model
    model_path = os.path.join(model_dir, "bert-base-chinese")
    config_path = os.path.join(model_dir, "config.json")
    model = RefineModel(tokenizer, model_path, config_path)

    for step, inputs in enumerate(dataloader):
        input_ids = inputs['input_ids']
        token_type_ids = inputs['token_type_ids']
        attention_mask = inputs['attention_mask']
        labels = inputs['labels']
        output = model(input_ids, attention_mask, token_type_ids)
        logits = output.argmax(dim=-1)

        print('input_ids=', input_ids)
        print('input_ids.shape=', input_ids.shape)

        print('token_type_ids=', token_type_ids)
        print('token_type_ids.shape=', token_type_ids.shape)

        print('attention_mask=', attention_mask)
        print('attention_mask.shape=', attention_mask.shape)

        print('labels=', labels)
        print('labels.shape=', labels.shape)

        print('output=', output)
        print('output.shape=', output.shape)

        print('logits=', logits)
        print('logits.shape=', logits.shape)

        break

# python3 run_pytorch.py ./zhwiki-latest-pages-articles_validation.txt ./bert-base-chinese ./input_data/ 384
if __name__ == '__main__':
    input_path = sys.argv[1]
    model_dir = sys.argv[2]
    save_dir = sys.argv[3]
    seq_length = int(sys.argv[4])
    batch_size = 1
    set_seed(42)

    preprocess(input_path, model_dir, save_dir, batch_size, seq_length)
