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
from model.ESRGAN import ESRGAN
import os
from glob import glob
import torch
from torchvision.utils import save_image
import torch.nn as nn
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')


class Tester:
    def __init__(self, config, data_loader):
        self.device = torch.device(f'npu:{NPU_CALCULATE_DEVICE}')
        self.checkpoint_dir = config.checkpoint_dir
        self.data_loader = data_loader
        self.scale_factor = config.scale_factor
        self.sample_dir = config.sample_dir
        self.num_epoch = config.num_epoch
        self.image_size = config.image_size
        self.upsampler = nn.Upsample(scale_factor=self.scale_factor)
        self.epoch = config.epoch
        self.build_model()

    def test(self):
        self.generator.eval()
        total_step = len(self.data_loader)
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)

        for step, image in enumerate(self.data_loader):
            low_resolution = image['lr'].to(f'npu:{NPU_CALCULATE_DEVICE}')
            high_resolution = image['hr'].to(f'npu:{NPU_CALCULATE_DEVICE}')
            fake_high_resolution = self.generator(low_resolution)
            low_resolution = self.upsampler(low_resolution)
            print(f"[Batch {step}/{total_step}]... ")

            result = torch.cat((low_resolution, fake_high_resolution, high_resolution), 2)
            save_image(result, os.path.join(self.sample_dir, f"SR_{step}.png"))

    def build_model(self):
        self.generator = ESRGAN(3, 3, 64, scale_factor=self.scale_factor).to(f'npu:{NPU_CALCULATE_DEVICE}')
        self.load_model()

    def load_model(self):
        print(f"[*] Load model from {self.checkpoint_dir}")
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if not os.listdir(self.checkpoint_dir):
            raise Exception(f"[!] No checkpoint in {self.checkpoint_dir}")

        generator = glob(os.path.join(self.checkpoint_dir, f'generator_{self.epoch - 1}.pth'))

        self.generator.load_state_dict(torch.load(generator[0]))