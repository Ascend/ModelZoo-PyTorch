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

conf_1p = {
    # please change to your own path
    "WORK_PATH": ".",
    "ASCEND_VISIBLE_DEVICES": "0",
    'profiling':'None',
    'start_step':-1,
    'stop_step':-1,
    "data": {
        'dataset_path': "../../CASIA-B-Pre/",
        'resolution': '64',
        'dataset': 'CASIA-B',
        # In CASIA-B, data of subject #5 is incomplete.
        # Thus, we ignore it in training.
        # For more detail, please refer to
        # function: utils.data_loader.load_data
        'pid_num': 73,
        'pid_shuffle': False,
        'device_num': 1,
    },
    "model": {
        'hidden_dim': 256,
        'lr': 8e-4,
        'hard_or_full_trip': 'full',
        'batch_size': (8, 16),
        'restore_iter': 0,
        'total_iter': 40000,
        'margin': 0.2,
        'num_workers': 8,
        'frame_num': 30,
        'model_name': 'GaitSet',
    },
}

conf_8p = {
    # please change to your own path
    "WORK_PATH": ".",
    "ASCEND_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
    'profiling':'None',
    'start_step':-1,
    'stop_step':-1,
    "data": {
        'dataset_path': "../../CASIA-B-Pre/",
        'resolution': '64',
        'dataset': 'CASIA-B',
        # In CASIA-B, data of subject #5 is incomplete.
        # Thus, we ignore it in training.
        # For more detail, please refer to
        # function: utils.data_loader.load_data
        'pid_num': 73,
        'pid_shuffle': False,
        'device_num': 8,
    },
    "model": {
        'hidden_dim': 256,
        'lr': 8e-4,
        'hard_or_full_trip': 'full',
        'batch_size': (8, 16),
        'restore_iter': 0,
        'total_iter': 40000,
        'margin': 0.2,
        'num_workers': 8,
        'frame_num': 30,
        'model_name': 'GaitSet',
    },
}
