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

import numpy as np
import torch
import torch.nn.functional as F
import os
import struct 


def bin2tensor(binName):
    size = os.path.getsize(binName)    
    binfile = open(binName, 'rb')   
    Len = int(size / 4)         
    res=[]          
    for i in range(Len):
        data = binfile.read(4)         
        num = struct.unpack('f', data)
        res.append(num[0])
    
    binfile.close()
    dim_res = np.array(res)
    dim_res = torch.from_numpy(dim_res)

    return dim_res


def mel_loss(mel_out, mel_tgt):
    """
    mel_out: torch.tensor, shape(batchsize, 80, 900)
    mel_tgt: torch.tensor, shape(batchsize, 80, 900)
    """
    mel_tgt = mel_tgt.transpose(1, 2)
    mel_out = mel_out.transpose(1, 2)

    mel_mask = mel_tgt.ne(0).float()
    mel_mask_sum = mel_mask.sum()

    loss_fn = F.mse_loss
    mel_loss = loss_fn(mel_out, mel_tgt, reduction='none')
    mel_loss = (mel_loss * mel_mask).sum() / mel_mask_sum

    return mel_loss


def test_om():
    tgt_path = "./test/mel_tgt_pth/"
    out_path = './test/result/dumpOutput_device0/'
    data_len = 100
    mel_loss_total = 0
    for i in range(data_len):
        mel_out = bin2tensor(os.path.join(out_path, f"data{i}.bin")).reshape(1, 80, 900)
        mel_tgt = torch.load(os.path.join(tgt_path, f"mel_tgt{i}.pth"))
        mel_loss_ = mel_loss(mel_out, mel_tgt)
        mel_loss_total += mel_loss_
    mel_loss_average = mel_loss_total / data_len
    print("mel_loss_average", mel_loss_average.item())

def test_pth():
    out_path = './test/mel_out_pth/'
    tgt_path = './test/mel_tgt_pth/'
    data_len = 100
    mel_loss_total = 0
    for i in range(data_len):
        mel_out = torch.load(os.path.join(out_path, f"mel_out{i}.pth"))
        mel_tgt = torch.load(os.path.join(tgt_path, f"mel_tgt{i}.pth"))
        mel_loss_ = mel_loss(mel_out, mel_tgt)
        mel_loss_total += mel_loss_
    mel_loss_average = mel_loss_total / data_len
    print("mel_loss_average", mel_loss_average.item())



if __name__ == "__main__":
    print("==================om==================")
    test_om()
    print("==================pth==================")
    test_pth()