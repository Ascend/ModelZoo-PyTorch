# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch.utils.data as data
import os.path
import random
import torch
from PIL import Image

class ImageAttributeDataset(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, filename, transform):
        """Initialize and preprocess the CelebA dataset."""
        self.lines = [line.rstrip().split() for line in open(filename, 'r')]
        self.transform = transform
        self.length = len(self.lines)

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        line = self.lines[index]
        image = Image.open(line[0])
        conditions = [int(condition) for condition in line[1:]]
        return self.transform(image), torch.Tensor(conditions)

    def __len__(self):
        """Return the number of images."""
        return self.length