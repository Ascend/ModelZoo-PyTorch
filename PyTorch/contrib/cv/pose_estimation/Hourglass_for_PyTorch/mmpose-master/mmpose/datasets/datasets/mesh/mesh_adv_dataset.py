# -*- coding: utf-8 -*-
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

import numpy as np
from torch.utils.data import Dataset

from mmpose.datasets.builder import DATASETS, build_dataset


@DATASETS.register_module()
class MeshAdversarialDataset(Dataset):
    """Mix Dataset for the adversarial training in 3D human mesh estimation
    task.

    The dataset combines data from two datasets and
    return a dict containing data from two datasets.

    Args:
        train_dataset (Dataset): Dataset for 3D human mesh estimation.
        adversarial_dataset (Dataset): Dataset for adversarial learning,
            provides real SMPL parameters.
    """

    def __init__(self, train_dataset, adversarial_dataset):
        super().__init__()
        self.train_dataset = build_dataset(train_dataset)
        self.adversarial_dataset = build_dataset(adversarial_dataset)
        self.length = len(self.train_dataset)

    def __len__(self):
        """Get the size of the dataset."""
        return self.length

    def __getitem__(self, i):
        """Given index, get the data from train dataset and randomly sample an
        item from adversarial dataset.

        Return a dict containing data from train and adversarial dataset.
        """
        data = self.train_dataset[i]
        ind_adv = np.random.randint(
            low=0, high=len(self.adversarial_dataset), dtype=np.int)
        data.update(self.adversarial_dataset[ind_adv %
                                             len(self.adversarial_dataset)])
        return data
