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

from torch.utils.data import Dataset as TorchDataset

from concern.config import Configurable, State


class SliceDataset(TorchDataset, Configurable):
    dataset = State()
    start = State()
    end = State()

    def __init__(self, **kwargs):
        self.load_all(**kwargs)

        if self.start is None:
            self.start = 0
        if self.end is None:
            self.end = len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[self.start + idx]

    def __len__(self):
        return self.end - self.start
