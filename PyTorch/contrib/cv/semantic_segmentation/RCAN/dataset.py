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
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import glob
import numpy as np
import PIL.Image as pil_image
from torchvision import transforms
import torch
from PIL import ImageFile


class Dataset(object):
    def __init__(self, images_dir, patch_size, scale, transform = None):
        self.image_files = sorted(glob.glob(images_dir + '/*.png'))
        self.patch_size = patch_size
        self.scale = scale
        self.transform = transform

    def __getitem__(self, idx):
        hr = pil_image.open(self.image_files[idx]).convert('RGB')
        if self.transform != None:
            hr = self.transform(hr)
        # if use origin datase
        # randomly crop patch from training set
        # crop_x = random.randint(0, hr.width - self.patch_size * self.scale)
        # crop_y = random.randint(0, hr.height - self.patch_size * self.scale)
        # hr = hr.crop((crop_x, crop_y, crop_x + self.patch_size * self.scale, crop_y + self.patch_size * self.scale))

        # degrade lr with Bicubic
        lr = hr.resize((self.patch_size, self.patch_size), resample=pil_image.BICUBIC)

        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)

        hr = np.transpose(hr, axes=[2, 0, 1])
        lr = np.transpose(lr, axes=[2, 0, 1])

        # normalization
        hr /= 255.0
        lr /= 255.0

        return lr, hr,self.image_files[idx]

    def __len__(self):
        return len(self.image_files)

class Dataset_test_label(object):
    def __init__(self, images_dir, scale,):
        self.image_files = sorted(glob.glob(images_dir + '/*.png'))
        self.scale = scale

    def __getitem__(self, idx):

        filename = os.path.basename(self.image_files[idx]).split('.')[0]

        hr = pil_image.open(self.image_files[idx]).convert('RGB')
        hr = hr.crop((0, 0, hr.width // self.scale * self.scale, hr.height // self.scale* self.scale))

        lr = hr.resize((hr.width // self.scale, hr.height // self.scale), resample=pil_image.BICUBIC)

        bicubic = lr.resize((hr.width, hr.height), resample=pil_image.BICUBIC)

        # bicubic = np.asarray(bicubic)
        bicubic = torch.from_numpy(np.array(bicubic))
        hr = torch.from_numpy(np.array(hr))

        lr = np.array(lr).astype(np.float32)
        lr = np.transpose(lr, axes=[2, 0, 1])
        lr /= 255.0

        return hr,  lr, bicubic, filename

    def __len__(self):
        return len(self.image_files)


