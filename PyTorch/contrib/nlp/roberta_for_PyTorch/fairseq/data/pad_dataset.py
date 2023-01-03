# Copyright 2021 Huawei Technologies Co., Ltd
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
# ============================================================================

from fairseq.data import data_utils

from . import BaseWrapperDataset


class PadDataset(BaseWrapperDataset):
    def __init__(self, dataset, pad_idx, pad_to_length, left_pad):
        super().__init__(dataset)
        self.pad_idx = pad_idx
        self.left_pad = left_pad
        self.pad_to_length = pad_to_length

    def collater(self, samples):
        return data_utils.collate_tokens(samples, self.pad_idx, left_pad=self.left_pad, pad_to_length=self.pad_to_length)


class LeftPadDataset(PadDataset):
    def __init__(self, dataset, pad_idx):
        super().__init__(dataset, pad_idx, left_pad=True)


class RightPadDataset(PadDataset):
    def __init__(self, dataset, pad_to_length, pad_idx):
        super().__init__(dataset, pad_idx, pad_to_length=pad_to_length, left_pad=False)
