# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------
# -----------------------------------------------------
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===========================================================================
import torch.nn as nn


class PixelUnshuffle(nn.Module):
    '''
    Initialize: inplanes, planes, upscale_factor
    OUTPUT: (planes // upscale_factor^2) * ht * wd
    '''

    def __init__(self, downscale_factor=2):
        super(PixelUnshuffle, self).__init__()
        self._r = downscale_factor

    def forward(self, x):
        b, c, h, w = x.shape
        out_c = c * (self._r * self._r)
        out_h = h // self._r
        out_w = w // self._r

        x_view = x.contiguous().view(b, c, out_h, self._r, out_w, self._r)
        x_prime = x_view.permute(0, 1, 3, 5, 2, 4).contiguous().view(b, out_c, out_h, out_w)

        return x_prime
