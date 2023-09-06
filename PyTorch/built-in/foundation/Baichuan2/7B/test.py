import json
import torch

import torch_npu
from torch_npu.contrib import transfer_to_npu

import transformers
from rich.table import Table
from rich.console import Console


class BaichuanClassifier:
    def __init__(
            self,
            model_path: str,
            prompt: str = '{review} -> '
    ):
        self.model_path = model_path
        self.prompt = prompt
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map='auto',
            trust_remote_code=True,
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_path,
            use_fast=False,
            trust_remote_code=True
        )
        self.user_token_id = 195
        self.assistant_token_id = 196
        self.pad_token_id = 0

    def __call__(self, review):
        prompt = self.prompt.format(review=review)
        input_ids = self.tokenizer.encode(prompt)
        input_ids = [self.user_token_id] + input_ids + [self.assistant_token_id]
        input_ids = torch.LongTensor([input_ids]).to(self.model.device)
        outputs = self.model.generate(input_ids, max_new_tokens=5)
        output_ids = outputs[0][len(input_ids[0]):]
        answer = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return answer


if __name__ == '__main__':
    torch.npu.set_compile_mode(jit_compile=True)

    # 原对话模型
    base_classifier = BaichuanClassifier('Baichuan2_7B')
    # 微调后的情感分类模型
    tune_classifier = BaichuanClassifier('outputs/checkpoint-312')

    table = Table()
    table = Table(title='模型对比：base对话模型 vs tune情感识别模型')
    table.add_column('review', style='magenta')
    table.add_column('label', style='cyan', justify='center')
    table.add_column('base', style='red')
    table.add_column('tune', style='green', justify='center')

    dataset = []
    for i, line in enumerate(open('data/eval.jsonl')):
        if i == 5:
            break
        data = json.loads(line.strip())
        base = base_classifier(data['review'])
        tune = tune_classifier(data['review'])
        table.add_row(data['review'], data['label'], base, tune)
    Console().print(table)
