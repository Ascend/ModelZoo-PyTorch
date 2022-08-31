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


conf_8p = {
    "WORK_PATH": "work",
    "CUDA_VISIBLE_DEVICES": "4,5,6,7",
    "data": {
        'dataset_path': "./../CASIA-B-Pre",
        'resolution': '64',
        'dataset': 'CASIA-B',
        'pid_num': 73,
        'pid_shuffle': False,
    },
    "model": {
        'hidden_dim': 256,
        'lr': 4e-5,
        'hard_or_full_trip': 'full',
        'batch_size': (8, 16),
        'restore_iter': 0,
        'total_iter': 80000,
        'margin': 0.2,
        'num_workers': 8,
        'frame_num': 30,
        'model_name': 'GaitSet',
    },
}
