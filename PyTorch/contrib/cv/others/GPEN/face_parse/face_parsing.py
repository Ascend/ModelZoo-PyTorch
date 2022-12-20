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
import torch.nn.functional as F

from parse_model import ParseNet


class FaceParse(object):
    def __init__(self, base_dir='./', model='ParseNet-latest', device='cuda'):
        self.faceparse = None
        self.mfile = os.path.join(base_dir, 'weights', model + '.pth')
        self.size = 512
        self.device = device

        '''
        0: 'background' 1: 'skin'   2: 'nose'
        3: 'eye_g'  4: 'l_eye'  5: 'r_eye'
        6: 'l_brow' 7: 'r_brow' 8: 'l_ear'
        9: 'r_ear'  10: 'mouth' 11: 'u_lip'
        12: 'l_lip' 13: 'hair'  14: 'hat'
        15: 'ear_r' 16: 'neck_l'    17: 'neck'
        18: 'cloth'
        '''
        self.MASK_COLORMAP = [0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 255, 255, 0]
        self.load_model()

    def load_model(self):
        self.faceparse = ParseNet(self.size, self.size, 32, 64, 19, norm_type='bn', relu_type='LeakyReLU',
                                  ch_range=[32, 256])
        self.faceparse.load_state_dict(torch.load(self.mfile, map_location=torch.device('cpu')))
        self.faceparse.to(self.device)
        self.faceparse.eval()

    def process(self, im):
        im = cv2.resize(im, (self.size, self.size))
        imt = self.img2tensor(im)
        pred_mask, _ = self.faceparse(imt)
        mask = self.tenor2mask(pred_mask)

        return mask

    def process_tensor(self, imt):
        imt = F.interpolate(imt.flip(1) * 2 - 1, (self.size, self.size))
        pred_mask, _ = self.faceparse(imt)

        mask = pred_mask.argmax(dim=1)
        for idx, color in enumerate(self.MASK_COLORMAP):
            mask = torch.where(mask == idx, color, mask)
        mask = mask.unsqueeze(0)

        return mask

    def img2tensor(self, img):
        img = img[..., ::-1]
        img = img / 255. * 2 - 1
        img_tensor = torch.from_numpy(np.float32(img.transpose(2, 0, 1))).unsqueeze(0).to(self.device)
        return img_tensor.float()

    def tenor2mask(self, tensor):
        if len(tensor.shape) < 4:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[1] > 1:
            tensor = tensor.argmax(dim=1)

        tensor = tensor.squeeze(1).data.cpu().numpy()
        color_maps = []
        for t in tensor:
            tmp_img = np.zeros(tensor.shape[1:])
            for idx, color in enumerate(self.MASK_COLORMAP):
                tmp_img[t == idx] = color
            color_maps.append(tmp_img.astype(np.uint8))
        return color_maps
