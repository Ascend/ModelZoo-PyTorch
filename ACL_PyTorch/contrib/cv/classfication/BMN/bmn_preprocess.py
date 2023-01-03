# Copyright 2022 Huawei Technologies Co., Ltd
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
import sys

import numpy as np
import torch
from tqdm import tqdm

import opts
sys.path.append(r"./BMN-Boundary-Matching-Network")
from dataset import VideoDataSet

def preprocess(opt):
    opt['mode'] = 'inference'
    bin_path = opt['save_dir']
    
    if not os.path.exists(bin_path):
        os.makedirs(bin_path)
    test_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="validation"),
                                              batch_size=1, shuffle=False,
                                              num_workers=8, pin_memory=True, drop_last=False)
    
    print('Preprocess started!')
    with torch.no_grad():
        for idx, input_data in tqdm(test_loader):
            input_data = input_data.detach().cpu().numpy()
            assert input_data.shape == (1, 400, 100)
            bin_name = '{:0>4d}.bin'.format(int(idx))
            input_data.tofile(os.path.join(bin_path, bin_name))
    print('Preprocess Finished!')

if __name__ == '__main__':
    option = opts.parse_opt()
    option = vars(option)
    preprocess(option)
    