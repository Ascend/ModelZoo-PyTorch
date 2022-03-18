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

from concern.config import State
from .data_process import DataProcess


class MakeCenterPoints(DataProcess):
    box_key = State(default='charboxes')
    size = State(default=32)

    def process(self, data):
        shape = data['image'].shape[:2]
        points = np.zeros((self.size, 2), dtype=np.float32)
        boxes = np.array(data[self.box_key])[:self.size]

        size = boxes.shape[0]
        points[:size] = boxes.mean(axis=1)
        data['points'] = (points / shape[::-1]).astype(np.float32)
        return data
