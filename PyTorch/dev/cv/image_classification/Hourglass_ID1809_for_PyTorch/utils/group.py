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
import numpy as np
import torch
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')

def match_format(dic):
    loc = dic['loc_k'][0,:,0,:]
    val = dic['val_k'][0,:,:]
    ans = np.hstack((loc, val))
    ans = np.expand_dims(ans, axis = 0) 
    ret = []
    ret.append(ans)
    return ret

class HeatmapParser:
    def __init__(self):
        from torch import nn
        self.pool = nn.MaxPool2d(3, 1, 1)

    def nms(self, det):
        maxm = self.pool(det)
        maxm = torch.eq(maxm, det).float()
        det = det * maxm
        return det

    def calc(self, det):
        with torch.no_grad():
            det = torch.autograd.Variable(torch.Tensor(det))
            # This is a better format for future version pytorch

        det = self.nms(det)
        h = det.size()[2]
        w = det.size()[3]
        det = det.view(det.size()[0], det.size()[1], -1)
        val_k, ind = det.topk(1, dim=2)

        x = ind % w
        y = (ind / w).long()
        ind_k = torch.stack((x, y), dim=3)
        ans = {'loc_k': ind_k, 'val_k': val_k}
        return {key:ans[key].cpu().data.numpy() for key in ans}

    def adjust(self, ans, det):
        for batch_id, people in enumerate(ans): 
            for people_id, i in enumerate(people):
                for joint_id, joint in enumerate(i):
                    if joint[2]>0:
                        y, x = joint[0:2]
                        xx, yy = int(x), int(y)
                        tmp = det[0][joint_id]
                        if tmp[xx, min(yy+1, tmp.shape[1]-1)]>tmp[xx, max(yy-1, 0)]:
                            y+=0.25
                        else:
                            y-=0.25

                        if tmp[min(xx+1, tmp.shape[0]-1), yy]>tmp[max(0, xx-1), yy]:
                            x+=0.25
                        else:
                            x-=0.25
                        ans[0][0, joint_id, 0:2] = (y+0.5, x+0.5)
        return ans

    def parse(self, det, adjust=True):
        ans = match_format(self.calc(det))
        if adjust:
            ans = self.adjust(ans, det)
        return ans
