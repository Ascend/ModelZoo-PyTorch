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

import numpy as np

from data.quad import Quad
from concern.config import State
from .data_process import DataProcess


class SerializeBox(DataProcess):
    box_key = State(default='charboxes')
    format = State(default='NP2')

    def process(self, data):
        data[self.box_key] = data['lines'].quads
        return data


class UnifyRect(SerializeBox):
    max_size = State(default=64)

    def process(self, data):
        h, w = data['image'].shape[:2]
        boxes = np.zeros((self.max_size, 4), dtype=np.float32)
        mask_has_box = np.zeros(self.max_size, dtype=np.float32)
        data = super().process(data)
        quad = data[self.box_key]
        assert quad.shape[0] <= self.max_size
        boxes[:quad.shape[0]] = quad.rectify() / np.array([w, h, w, h]).reshape(1, 4)
        mask_has_box[:quad.shape[0]] = 1.
        data['boxes'] = boxes
        data['mask_has_box'] = mask_has_box
        return data
