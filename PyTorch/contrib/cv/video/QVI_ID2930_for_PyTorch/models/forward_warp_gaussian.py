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
# ============================================================================
# class WarpLayer warps image x based on optical flow flo.
import time

import torch
import torch.nn as nn
import numpy


class ForwardWarp(nn.Module):
    """docstring for WarpLayer"""
    def __init__(self,):
        super(ForwardWarp, self).__init__()

    def forward(self, img, flo):
        """
            -img: image (N, C, H, W)
            -flo: optical flow (N, 2, H, W)
            elements of flo is in [0, H] and [0, W] for dx, dy

        """
        # (x1, y1)		(x1, y2)
        # +---------------+
        # |				  |
        # |	o(x, y) 	  |
        # |				  |
        # |				  |
        # |				  |
        # |				  |
        # +---------------+
        # (x2, y1)		(x2, y2)
        # _1 = time.time()
        N, C, _, _ = img.size()
        # translate start-point optical flow to end-point optical flow
        y = flo[:, 0:1:, :]
        x = flo[:, 1:2, :, :]

        x = x.repeat(1, C, 1, 1)
        y = y.repeat(1, C, 1, 1)

        # Four point of square (x1, y1), (x1, y2), (x2, y1), (y2, y2)
        x1 = torch.floor(x)
        x2 = x1 + 1
        y1 = torch.floor(y)
        y2 = y1 + 1

        # firstly, get gaussian weights
        w11, w12, w21, w22 = self.get_gaussian_weights(x, y, x1, x2, y1, y2)
        # secondly, sample each weighted corner
        img11, o11 = self.sample_one(img, x1, y1, w11)
        img12, o12 = self.sample_one(img, x1, y2, w12)
        img21, o21 = self.sample_one(img, x2, y1, w21)
        img22, o22 = self.sample_one(img, x2, y2, w22)
        imgw = img11 + img12 + img21 + img22
        o = o11 + o12 + o21 + o22
        # print("ForwardWarp", time.time() - _1)
        return imgw, o

    def get_gaussian_weights(self, x, y, x1, x2, y1, y2):
        w11 = torch.exp(-((x - x1)**2 + (y - y1)**2))
        w12 = torch.exp(-((x - x1)**2 + (y - y2)**2))
        w21 = torch.exp(-((x - x2)**2 + (y - y1)**2))
        w22 = torch.exp(-((x - x2)**2 + (y - y2)**2))
        return w11, w12, w21, w22

    def sample_one(self, img, shiftx, shifty, weight):
        """
        Input:
            -img (N, C, H, W)
            -shiftx, shifty (N, c, H, W)
        """
        N, C, H, W = img.size()
        # flatten all (all restored as Tensors)
        flat_shiftx = shiftx.view(-1)
        flat_shifty = shifty.view(-1)
        flat_basex = torch.arange(0, H, requires_grad=False, device=img.device, dtype=torch.int).view(-1, 1)[None, None].repeat(N, C, 1, W).view(-1)
        flat_basey = torch.arange(0, W, requires_grad=False, device=img.device, dtype=torch.int).view(1, -1)[None, None].repeat(N, C, H, 1).view(-1)
        flat_weight = weight.view(-1)
        flat_img = img.view(-1)
        # The corresponding positions in I1
        idxn = torch.arange(0, N, requires_grad=False, device=img.device, dtype=torch.int).view(N, 1, 1, 1).repeat(1, C, H, W).view(-1)
        idxc = torch.arange(0, C, requires_grad=False, device=img.device, dtype=torch.int).view(1, C, 1, 1).repeat(N, 1, H, W).view(-1)
        # ttype = flat_basex.type()
        idxx = flat_shiftx.int() + flat_basex
        idxy = flat_shifty.int() + flat_basey
        # recording the inside part the shifted
        mask = idxx.ge(0) & idxx.lt(H) & idxy.ge(0) & idxy.lt(W)
        # Mask off points out of boundaries
        ids = (idxn*C*H*W + idxc*H*W + idxx*W + idxy)
        ids_mask = torch.masked_select(ids, mask).to(img.device)
        img_warp = torch.zeros([N*C*H*W, ], device=img.device)
        img_warp.put_(ids_mask, torch.masked_select(flat_img*flat_weight, mask).float(), accumulate=True)
        one_warp = torch.zeros([N*C*H*W, ], device=img.device)
        one_warp.put_(ids_mask, torch.masked_select(flat_weight, mask).float(), accumulate=True)
        return img_warp.view(N, C, H, W), one_warp.view(N, C, H, W)
