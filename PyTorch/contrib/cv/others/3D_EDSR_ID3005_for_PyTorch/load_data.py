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
from torch.utils.data import Dataset
from os import listdir
from skimage import io
from os.path import join
import numpy as np
from torchvision.transforms import Compose, ToTensor


# training/testing data generation
class DatasetFromFolder(Dataset):
    def __init__(self, root):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames1 = [join(root, x) for x in listdir(root)][0]  # LR
        self.image_filenames2 = [join(root, x) for x in listdir(root)][1]  # HR
        if "HR" in self.image_filenames1:
            tmp = self.image_filenames1
            self.image_filenames1 = self.image_filenames2
            self.image_filenames2 = tmp
        LIST1 = listdir(self.image_filenames1)
        LIST2 = listdir(self.image_filenames2)
        LIST1.sort(key=lambda x: int(x[:-4]))
        LIST2.sort(key=lambda x: int(x[:-4]))
        self.img1 = [join(self.image_filenames1, x) for x in LIST1]
        self.img2 = [join(self.image_filenames2, x) for x in LIST2]
        self.train_hr_transform = tensor_transform()
        self.train_lr_transform = tensor_transform()

    def __getitem__(self, index):
        lr_image = self.train_lr_transform(io.imread(self.img1[index]))  # LR
        hr_image = self.train_lr_transform(io.imread(self.img2[index]))  # HR
        hr_image = np.expand_dims(hr_image, axis=0)
        lr_image = np.expand_dims(lr_image, axis=0)
        return lr_image, hr_image

    def __len__(self):
        return len(self.img1)


def tensor_transform():
    return Compose([
        ToTensor(),
    ])
