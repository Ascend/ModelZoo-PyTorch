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
import glob
import math
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import degradations


class GfpganDegradation(object):
    def __init__(self):
        self.kernel_list = ['iso', 'aniso']
        self.kernel_prob = [0.5, 0.5]
        self.blur_kernel_size = 41
        self.blur_sigma = [0.1, 10]
        self.downsample_range = [0.8, 8]
        self.noise_range = [0, 20]
        self.jpeg_range = [60, 100]
        self.gray_prob = 0.2
        self.color_jitter_prob = 0.0
        self.color_jitter_pt_prob = 0.0
        self.shift = 20 / 255.

    def degrade_process(self, img_gt):
        if random.random() > 0.5:
            img_gt = cv2.flip(img_gt, 1)

        h, w = img_gt.shape[:2]

        # random color jitter 
        if np.random.uniform() < self.color_jitter_prob:
            jitter_val = np.random.uniform(-self.shift, self.shift, 3).astype(np.float32)
            img_gt = img_gt + jitter_val
            img_gt = np.clip(img_gt, 0, 1)

            # random grayscale
        if np.random.uniform() < self.gray_prob:
            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
            img_gt = np.tile(img_gt[:, :, None], [1, 1, 3])

        # ------------------------ generate lq image ------------------------ #
        # blur
        kernel = degradations.random_mixed_kernels(
            self.kernel_list,
            self.kernel_prob,
            self.blur_kernel_size,
            self.blur_sigma,
            self.blur_sigma, [-math.pi, math.pi],
            noise_range=None)
        img_lq = cv2.filter2D(img_gt, -1, kernel)
        # downsample
        scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
        img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)

        # noise
        if self.noise_range is not None:
            img_lq = degradations.random_add_gaussian_noise(img_lq, self.noise_range)
        # jpeg compression
        if self.jpeg_range is not None:
            img_lq = degradations.random_add_jpg_compression(img_lq, self.jpeg_range)

        # round and clip
        img_lq = np.clip((img_lq * 255.0).round(), 0, 255) / 255.

        # resize to original size
        img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)

        return img_gt, img_lq


class FaceDataset(Dataset):
    def __init__(self, path, resolution=512):
        self.resolution = resolution

        self.HQ_imgs = glob.glob(os.path.join(path, '*.*'))
        self.length = len(self.HQ_imgs)

        self.degrader = GfpganDegradation()

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img_gt = cv2.imread(self.HQ_imgs[index], cv2.IMREAD_COLOR)
        img_gt = cv2.resize(img_gt, (self.resolution, self.resolution), interpolation=cv2.INTER_AREA)

        # BFR degradation
        # We adopt the degradation of GFPGAN for simplicity, which however differs from our implementation in the paper.
        # Data degradation plays a key role in BFR. Please replace it with your own methods.
        img_gt = img_gt.astype(np.float32) / 255.
        img_gt, img_lq = self.degrader.degrade_process(img_gt)

        img_gt = (torch.from_numpy(img_gt) - 0.5) / 0.5
        img_lq = (torch.from_numpy(img_lq) - 0.5) / 0.5

        img_gt = img_gt.permute(2, 0, 1).flip(0)  # BGR->RGB
        img_lq = img_lq.permute(2, 0, 1).flip(0)  # BGR->RGB

        return img_lq, img_gt
