import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Baichuan2_7B_2T8_V1")
    max_seq_length: Optional[int] = field(default=256)


@dataclass
class DataArguments:
    train_data: str = field(metadata={"help": "Path to the training data."})
    eval_data: str = field(metadata={"help": "Path to the evaluation data."})


class ReviewDataset(Dataset):
    def __init__(
            self,
            data_file: str,
            max_seq_length: int,
            tokenizer: transformers.PreTrainedTokenizer,
            prompt: str = '{review} -> '):
        super().__init__()
        self.data_file = data_file
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.prompt = prompt
        self._preprocess()

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    def _preprocess(self):
        logging.info(f"loading {self.data_file}")
        self.dataset = []
        max_length = 0
        for i, line in enumerate(open(self.data_file)):
            data = json.loads(line.strip())
            prompt = self.prompt.format(review=data['review'])
            input_ids = [195] + self.tokenizer.encode(prompt) + [196]
            output_ids = self.tokenizer.encode(data['label']) + [self.tokenizer.eos_token_id]
            valid_len = len(input_ids) + len(output_ids)
            if valid_len > self.max_seq_length:
                continue
            pad_ids = [self.tokenizer.pad_token_id] * (self.max_seq_length - valid_len)
            self.dataset.append({
                'input_ids': torch.LongTensor(input_ids + output_ids + pad_ids),
                'labels': torch.LongTensor([-100] * len(input_ids) + output_ids + [-100] * len(pad_ids))
            })
            max_length = max(max_length, len(self.dataset[-1]['input_ids']))
            if i == 0:
                logging.info({'prompt': prompt, 'label': data['label']})
                logging.info(self.dataset[0])
        logging.info(f"load #{len(self.dataset)} sample from {self.data_file}, max_length={max_length}")


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, transformers.TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False,
        trust_remote_code=True
    )

    trainer = Trainer(
        args=training_args,
        model=model,
        tokenizer=tokenizer,
        train_dataset=ReviewDataset(data_args.train_data, model_args.max_seq_length, tokenizer),
        eval_dataset=ReviewDataset(data_args.eval_data, model_args.max_seq_length, tokenizer),
        data_collator=DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
