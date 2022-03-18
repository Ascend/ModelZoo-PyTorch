# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from data import common

import cv2
import numpy as np
import imageio

import torch
import torch.utils.data as data

class Video(data.Dataset):
    def __init__(self, args, name='Video', train=False, benchmark=False):
        self.args = args
        self.name = name
        self.scale = args.scale
        self.idx_scale = 0
        self.train = False
        self.do_eval = False
        self.benchmark = benchmark

        self.filename, _ = os.path.splitext(os.path.basename(args.dir_demo))
        self.vidcap = cv2.VideoCapture(args.dir_demo)
        self.n_frames = 0
        self.total_frames = int(self.vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __getitem__(self, idx):
        success, lr = self.vidcap.read()
        if success:
            self.n_frames += 1
            lr, = common.set_channel(lr, n_channels=self.args.n_colors)
            lr_t, = common.np2Tensor(lr, rgb_range=self.args.rgb_range)

            return lr_t, -1, '{}_{:0>5}'.format(self.filename, self.n_frames)
        else:
            vidcap.release()
            return None

    def __len__(self):
        return self.total_frames

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale

