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
import os, sys
from tqdm import tqdm
from torch.utils.data import SequentialSampler, DataLoader
from transformers import BertTokenizerFast, DataCollatorForLanguageModeling, set_seed
from datasets import load_dataset


def tokenize_process(dataset, tokenizer_name, max_seq_length):
    tokenizer_kwargs = {
        'cache_dir': None,
        'use_feat': None,
        'revision': 'main',
        'use_auth_token': None
    }
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name, **tokenizer_kwargs)
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

    for step, inputs in enumerate(dataloader):
        for input_name in inputs:
            save_path = os.path.join(save_dir, input_name, "{}.bin".format(step))
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            inputs[input_name].numpy().tofile(save_path)


if __name__ == '__main__':
    input_path = sys.argv[1]
    model_dir = sys.argv[2]
    save_dir = sys.argv[3]
    seq_length = int(sys.argv[4])
    batch_size = 1
    set_seed(42)

    preprocess(input_path, model_dir, save_dir, batch_size, seq_length)
