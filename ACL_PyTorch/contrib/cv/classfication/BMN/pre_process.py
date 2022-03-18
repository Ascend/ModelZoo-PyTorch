# -*- coding: utf-8 -*-
# Copyright (c) Soumith Chintala 2016,
# All rights reserved
#
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://spdx.org/licenses/BSD-3-Clause.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

from dataset import VideoDataSet
import os
import torch
import numpy as np
import opts


opt = opts.parse_opt()
opt = vars(opt)

if __name__ == '__main__':
    opt['mode'] = 'inference'
    bin_path = 'preprocessed_imgs'
    
    if not os.path.exists(bin_path):
        os.makedirs(bin_path)
    test_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="validation"),
                                              batch_size=1, shuffle=False,
                                              num_workers=8, pin_memory=True, drop_last=False)
    tscale = opt["temporal_scale"]
    
    print('Preprocess started!')
    with torch.no_grad():
        for idx, input_data in test_loader:
            input_data = input_data.detach().cpu().numpy()
            assert input_data.shape == (1, 400, 100)
            bin_name = '{:0>4d}.bin'.format(int(idx))
            input_data.tofile(os.path.join(bin_path, bin_name))
    print('Preprocess Finished!')
