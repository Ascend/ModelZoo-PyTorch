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
import torch

from transformers import PreTrainedModel

from .custom_configuration import CustomConfig, NoSuperInitConfig


class CustomModel(PreTrainedModel):
    config_class = CustomConfig

    def __init__(self, config):
        super().__init__(config)
        self.linear = torch.nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x):
        return self.linear(x)

    def _init_weights(self, module):
        pass


class NoSuperInitModel(PreTrainedModel):
    config_class = NoSuperInitConfig

    def __init__(self, config):
        super().__init__(config)
        self.linear = torch.nn.Linear(config.attribute, config.attribute)

    def forward(self, x):
        return self.linear(x)

    def _init_weights(self, module):
        pass
