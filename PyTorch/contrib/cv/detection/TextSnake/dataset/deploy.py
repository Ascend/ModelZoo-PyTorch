# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================
import scipy.io as io
import numpy as np
import os

from dataset.data_util import pil_load_img
from dataset.dataload import TextDataset, TextInstance

class DeployDataset(TextDataset):

    def __init__(self, image_root, transform=None):
        super().__init__(transform)
        self.image_root = image_root
        self.image_list = os.listdir(image_root)

    def __getitem__(self, item):

        # Read image data
        image_id = self.image_list[item]
        image_path = os.path.join(self.image_root, image_id)
        image = pil_load_img(image_path)

        return self.get_test_data(image, image_id=image_id, image_path=image_path)

    def __len__(self):
        return len(self.image_list)

if __name__ == '__main__':
    import os
    from util.augmentation import BaseTransform, Augmentation
    from torch.utils.data import DataLoader

    means = (0.485, 0.456, 0.406)
    stds = (0.229, 0.224, 0.225)

    transform = BaseTransform(
        size=512, mean=means, std=stds
    )

    trainset = DeployDataset(
        image_root='data/total-text/Images/Train',
        transform=transform
    )

    loader = DataLoader(trainset, batch_size=1, num_workers=0)

    for img, meta in loader:
        print(img.size(), type(img))