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

import random
import torch


class PromptSpell(torch.nn.Module):
    def __init__(self, spell_length, hidden_size, spell_func):
        super(PromptSpell, self).__init__()
        self.spell_length = spell_length
        self.hidden_size = hidden_size
        self.spell_embeddings = torch.nn.Embedding(self.spell_length, self.hidden_size)
        self.spell_func = spell_func
        if self.spell_func == "lstm":
            self.lstm_head = torch.nn.LSTM(input_size=self.hidden_size,
                                           hidden_size=self.hidden_size,
                                           num_layers=2,
                                           # dropout=self.lstm_dropout,
                                           bidirectional=True,
                                           batch_first=True)  # .to(torch.device("cuda"))
            self.mlp_head = torch.nn.Sequential(torch.nn.Linear(2 * self.hidden_size, self.hidden_size),
                                                torch.nn.ReLU(),
                                                torch.nn.Linear(self.hidden_size, self.hidden_size))
        elif self.spell_func == "mlp":
            self.mlp_head = torch.nn.Sequential(torch.nn.Linear(self.hidden_size, self.hidden_size),
                                                torch.nn.ReLU(),
                                                torch.nn.Linear(self.hidden_size, self.hidden_size))
        elif self.spell_func != "none":
            raise NotImplementedError("Prompt function " + self.spell_func)

    def init_embedding(self, word_embeddings=None, task_tokens=None):
        num_words = 5000
        with torch.no_grad():
            for i in range(self.spell_length):
                rand_token = random.randrange(num_words)
                if task_tokens is None:
                    target_embedding = word_embeddings[rand_token]
                else:
                    word_embedding = word_embeddings[rand_token]
                    task_token = random.choice(task_tokens)
                    task_embedding = word_embeddings[task_token]
                    ratio = random.random()
                    target_embedding = word_embedding * ratio + task_embedding * (1 - ratio)
                self.spell_embeddings.weight.data[i] = target_embedding

    def forward(self):
        prompt_embeds = self.spell_embeddings.weight.unsqueeze(0)
        if self.spell_func == "lstm":
            prompt_embeds = self.lstm_head(prompt_embeds)[0]
        if self.spell_func == "lstm" or self.spell_func == "mlp":
            prompt_embeds = self.mlp_head(prompt_embeds)
        return prompt_embeds
