# coding=utf-8
# Copyright 2023 Huawei Technologies Co., Ltd
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

import os
import glob
import re
import json
import torch
import torch.utils.data
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

CHECKPOINT= "THUDM/chatglm2-6b"
DATA_PATH="./CEval"

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
model = AutoModel.from_pretrained(CHECKPOINT, trust_remote_code=True).bfloat16().cuda()

choices = ["A", "B", "C", "D"]
choice_tokens = [tokenizer.encode(choice, add_special_tokens=False)[0] for choice in choices]


def build_prompt(text):
    return "[Round {}]\n\n问：{}\n\n答：".format(1, text)


extraction_prompt = '综上所述，ABCD中正确的选项是：'

accuracy_dict, count_dict = {}, {}
with torch.no_grad():
    for entry in glob.glob(DATA_PATH, recursive=True):
        dataset = []
        with open(entry, encoding='utf-8') as file:
            for line in file:
                dataset.append(json.loads(line))
        correct = 0
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)
        for batch in tqdm(dataloader):
            texts = batch["inputs_pretokenized"]
            queries = [build_prompt(query) for query in texts]
            inputs = tokenizer(queries, padding=True, return_tensors="pt", truncation=True, max_length=2048).to('cuda')
            outputs = model.generate(**inputs, do_sample=False, max_new_tokens=512)
            intermediate_outputs = []
            for idx in range(len(outputs)):
                output = outputs.tolist()[idx][len(inputs["input_ids"][idx]):]
                response = tokenizer.decode(output)
                intermediate_outputs.append(response)
            answer_texts = [text + intermediate + "\n" + extraction_prompt for text, intermediate in
                            zip(texts, intermediate_outputs)]
            input_tokens = [build_prompt(answer_text) for answer_text in answer_texts]
            inputs = tokenizer(input_tokens, padding=True, return_tensors="pt", truncation=True, max_length=2048).to('cuda')
            outputs = model(**inputs, return_last_logit=True)
            logits = outputs.logits[:, -1]
            logits = logits[:, choice_tokens]
            preds = logits.argmax(dim=-1)
            correct += (preds.cpu() == batch["label"]).sum().item()
        accuracy = correct / len(dataset)
        print(entry, accuracy)
        accuracy_dict[entry] = accuracy
        count_dict[entry] = len(dataset)

acc_total, count_total = 0.0, 0
for key in accuracy_dict:
    acc_total += accuracy_dict[key] * count_dict[key]
    count_total += count_dict[key]
print(acc_total / count_total)