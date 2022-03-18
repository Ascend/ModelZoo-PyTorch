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

import numpy as np
import os
import sys
import torch
import torch.nn.functional as F

from DeepRL.deep_rl import *

def to_np(t):
    return t.cpu().detach().numpy()


def get_om_action(filename):
    np_state = np.loadtxt(filename)
    prediction = torch.from_numpy(np_state)
    prediction = F.softmax(prediction, dim=-1)
    atoms = torch.Tensor([-10.0000, -9.6000, -9.2000, -8.8000, -8.4000, -8.0000, -7.6000,
                          -7.2000, -6.8000, -6.4000, -6.0000, -5.6000, -5.2000, -4.8000,
                          -4.4000, -4.0000, -3.6000, -3.2000, -2.8000, -2.4000, -2.0000,
                          -1.6000, -1.2000, -0.8000, -0.4000, 0.0000, 0.4000, 0.8000,
                          1.2000, 1.6000, 2.0000, 2.4000, 2.8000, 3.2000, 3.6000,
                          4.0000, 4.4000, 4.8000, 5.2000, 5.6000, 6.0000, 6.4000,
                          6.8000, 7.2000, 7.6000, 8.0000, 8.4000, 8.8000, 9.2000,
                          9.6000, 10.0000])
    q = (prediction * atoms).sum(-1)
    off_action = to_np(q.argmax(-1))
    return off_action


def get_pth_action(filename):
    action = torch.load(filename)
    myaction = np.array(action).astype(np.int32)
    return myaction[0]


if __name__ == "__main__":
    action_file = sys.argv[1]
    out_file = sys.argv[2]
    num = int(sys.argv[3])
    out_dir = os.listdir(out_file)
    for out_dir_file in out_dir:
        om_filelist = os.listdir('{0}/{1}'.format(out_file, out_dir_file))
    file_num = len(om_filelist)
    equal = 0
    for i in range(file_num):
        pth_action = get_pth_action('{0}/{1}.pt'.format(action_file, i))
        om_action = get_om_action('{0}/{1}/{2}_output_0.txt'.format(out_file, out_dir_file, i))
        if pth_action==om_action:
            equal += 1
    print('om离线推理的精度是在线推理的：{0}'.format((equal/num)))
    if(equal > 0.9*num):
        print("Accuancy: OK")
    else:
        print("Accuancy: Failed")
