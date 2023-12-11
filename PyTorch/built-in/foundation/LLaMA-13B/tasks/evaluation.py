# coding=utf-8
# Copyright (c) 2023, HUAWEI CORPORATION. All rights reserved.
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
import logging
import json
import pandas as pd
import tqdm
from tasks.chat import Chat,load

MMLU_TEMPLATE_DIR = "tasks/mmlu.template.json"
class MmluEval:
    def __init__(self, test_dir, model_path , device , num_gpus ,instruction_template = "{few_shot_examples}\n\n"
                                      "{question}\nAnswer:",
                 output_template1 = r".*(?P<answer>[A|B|C|D])\..*",
                 output_template2 = r"(?P<answer>[A|B|C|D])"):
        self.test_dir = test_dir
        self.instruction_template = instruction_template
        self.output_template = [output_template1, output_template2]
        self.model_path = model_path
        self.device = device
        self.num_gpus = num_gpus
    def eval(self):
        answer_result = {}
        score_datas = []
        total_acc_n = 0
        total_n = 0
        rank = None
        model,tokenizer = load(self.model_path,self.device,self.num_gpus)
        with open(MMLU_TEMPLATE_DIR, encoding = 'utf-8') as f:
            mmlu_few_shot_template = json.load(f)
        for file in tqdm.tqdm(os.listdir(self.test_dir)):
            file_path = os.path.join(self.test_dir, file)
            data_df = pd.read_csv(file_path, names=['question', 'A', 'B', 'C', 'D', 'answer'])
            subject_name = file[0: -9]  # 文件命名规则是  {subject}_test.csv
            subject = subject_name.replace("_", " ")
            subject_result = []
            acc_n = 0
            for  idx, row in data_df.iterrows():
                test_question = f"{row['question']}\nA. {row['A']}\nB. {row['B']}\nC. {row['C']}\nD. {row['D']}"
                instruction = self.instruction_template.format(few_shot_examples = mmlu_few_shot_template[subject_name],
                                                               subject = subject,
                                                               question = test_question)
                result, rank = Chat(model , tokenizer , self.model_path , self.device , instruction)
                subject_result.append(result)
                if result:
                    for template in self.output_template:
                        if result[0] == row['answer']:
                            acc_n += 1
                            break
            if rank == 0:
                total_n += len(data_df)
                total_acc_n += acc_n
                answer_result[subject_name] = subject_result
                score_datas.append([subject_name, len(data_df), acc_n / len(data_df)])
        if rank == 0:
            score_datas.append(["total", total_n, total_acc_n / total_n])
        score_df = pd.DataFrame(columns = ['subject', 'question_n', 'acc'], data = score_datas)
        return answer_result, score_df
