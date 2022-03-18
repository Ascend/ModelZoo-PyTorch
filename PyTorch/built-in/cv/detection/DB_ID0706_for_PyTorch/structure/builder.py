#     Copyright [yyyy] [name of copyright owner]
#     Copyright 2020 Huawei Technologies Co., Ltd
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
#

from collections import OrderedDict

import torch

import structure.model
from concern.config import Configurable, State


class Builder(Configurable):
    model = State()
    model_args = State()

    def __init__(self, cmd={}, **kwargs):
        self.load_all(**kwargs)
        if 'backbone' in cmd:
            self.model_args['backbone'] = cmd['backbone']

    @property
    def model_name(self):
        return self.model + '-' + getattr(structure.model, self.model).model_name(self.model_args)

    def build(self, device, distributed=False, local_rank: int = 0):
        Model = getattr(structure.model,self.model)
        model = Model(self.model_args, device,
                      distributed=distributed, local_rank=local_rank)
        return model

