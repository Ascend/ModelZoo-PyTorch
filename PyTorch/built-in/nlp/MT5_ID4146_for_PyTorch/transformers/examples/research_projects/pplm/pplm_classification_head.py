# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from torch import nn


class ClassificationHead(nn.Module):
    """Classification Head for  transformer encoders"""

    def __init__(self, class_size, embed_size):
        super().__init__()
        self.class_size = class_size
        self.embed_size = embed_size
        # self.mlp1 = nn.Linear(embed_size, embed_size)
        # self.mlp2 = (nn.Linear(embed_size, class_size))
        self.mlp = nn.Linear(embed_size, class_size)

    def forward(self, hidden_state):
        # hidden_state = nn.functional.relu(self.mlp1(hidden_state))
        # hidden_state = self.mlp2(hidden_state)
        logits = self.mlp(hidden_state)
        return logits
