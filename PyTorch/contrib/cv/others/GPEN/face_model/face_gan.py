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

import cv2
import numpy as np
import torch

from gpen_model import FullGenerator


class FaceGAN(object):
    def __init__(self, base_dir='./', in_size=512, out_size=None, model=None, channel_multiplier=2, narrow=1, key=None,
                 is_norm=True, device='cuda'):
        self.model = None
        self.mfile = os.path.join(base_dir, 'weights', model+'.pth')
        self.n_mlp = 8
        self.device = device
        self.is_norm = is_norm
        self.in_resolution = in_size
        self.out_resolution = in_size if out_size is None else out_size
        self.key = key
        self.load_model(channel_multiplier, narrow)

    def load_model(self, channel_multiplier=2, narrow=1):
        if self.in_resolution == self.out_resolution:
            self.model = FullGenerator(self.in_resolution, 512, self.n_mlp, channel_multiplier,
                                       narrow=narrow, device=self.device)
        else:
            self.model = FullGenerator(self.in_resolution, self.out_resolution, 512, self.n_mlp,
                                       channel_multiplier, narrow=narrow, device=self.device)
        pretrained_dict = torch.load(self.mfile, map_location=torch.device('cpu'))
        print(pretrained_dict.keys())
        if self.key is not None: 
            pretrained_dict = pretrained_dict[self.key]
        self.model.load_state_dict(pretrained_dict['g'])
        self.model.to(self.device)
        self.model.eval()

    def process(self, img):
        img = cv2.resize(img, (self.in_resolution, self.in_resolution))
        img_t = self.img2tensor(img)

        with torch.no_grad():
            out, _ = self.model(img_t)
        del img_t

        out = self.tensor2img(out)

        return out

    def img2tensor(self, img):
        img_t = torch.from_numpy(img).to(self.device)/255.
        if self.is_norm:
            img_t = (img_t - 0.5) / 0.5
        img_t = img_t.permute(2, 0, 1).unsqueeze(0).flip(1)
        return img_t

    def tensor2img(self, img_t, pmax=255.0, imtype=np.uint8):
        if self.is_norm:
            img_t = img_t * 0.5 + 0.5
        img_t = img_t.squeeze(0).permute(1, 2, 0).flip(2)
        img_np = np.clip(img_t.float().cpu().numpy(), 0, 1) * pmax

        return img_np.astype(imtype)
