#
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
#
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from PIL import Image
import os
#import torchvision.transforms.functional as TF
from random import random
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')


class Datasets(Dataset):
    def __init__(self, image_size, scale):
        self.image_size = image_size
        self.scale = scale

        if not os.path.exists('datasets'):
            raise Exception(f"[!] dataset is not exited")

        self.image_file_name = sorted(os.listdir(os.path.join('./datasets', 'hr')))

    def __getitem__(self, item):
        file_name = self.image_file_name[item]
        high_resolution = Image.open(os.path.join('./datasets', 'hr', file_name)).convert('RGB')
        low_resolution = Image.open(os.path.join('./datasets', 'lr', file_name)).convert('RGB')

        if random() > 0.5:
            high_resolution = TF.vflip(high_resolution)
            low_resolution = TF.vflip(low_resolution)

        if random() > 0.5:
            high_resolution = TF.hflip(high_resolution)
            low_resolution = TF.hflip(low_resolution)

        high_resolution = TF.to_tensor(high_resolution)
        low_resolution = TF.to_tensor(low_resolution)

        images = {'lr': low_resolution, 'hr': high_resolution}

        return images

    def __len__(self):
        return len(self.image_file_name)
