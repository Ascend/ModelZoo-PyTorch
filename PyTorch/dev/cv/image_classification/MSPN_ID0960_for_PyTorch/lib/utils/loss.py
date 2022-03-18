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
"""
@author: Wenbo Li
@contact: fenglinglwb@gmail.com
"""

import torch
import torch.nn as nn

class JointsL2Loss(nn.Module):
    def __init__(self, has_ohkm=False, topk=8, thresh1=1, thresh2=0):
        super(JointsL2Loss, self).__init__()
        self.has_ohkm = has_ohkm
        self.topk = topk
        self.t1 = thresh1
        self.t2 = thresh2
        method = 'none' if self.has_ohkm else 'mean'
        self.calculate = nn.MSELoss(reduction=method)

    def forward(self, output, valid, label):
        assert output.shape == label.shape
        batch_size = output.size(0)
        keypoint_num = output.size(1)
        loss = 0

        for i in range(batch_size):
            pred = output[i].reshape(keypoint_num, -1)
            gt = label[i].reshape(keypoint_num, -1)

            if not self.has_ohkm:
                weight = torch.gt(valid[i], self.t1).float()
                gt = gt * weight 

            tmp_loss = self.calculate(pred, gt)

            if self.has_ohkm:
                tmp_loss = tmp_loss.mean(dim=1) 
                weight = torch.gt(valid[i].squeeze(), self.t2).float()
                tmp_loss = tmp_loss * weight 
                topk_val, topk_id = torch.topk(tmp_loss, k=self.topk, dim=0,
                        sorted=False)
                sample_loss = topk_val.mean(dim=0)
            else:
                sample_loss = tmp_loss

            loss = loss + sample_loss 

        return loss / batch_size


if __name__ == '__main__':
    a = torch.ones(1, 17, 12, 12)
    b = torch.ones(1, 17, 12, 12)
    c = torch.ones(1, 17, 1) * 2
    loss = JointsL2Loss()
    # loss = JointsL2Loss(has_ohkm=True)
    device = torch.device('npu')
    a = a.to(device)
    b = b.to(device)
    c = c.to(device)
    loss = loss.to(device)
    res = loss(a, c, b)
    print(res)


