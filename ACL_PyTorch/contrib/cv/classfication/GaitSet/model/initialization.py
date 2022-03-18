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


import os
from copy import deepcopy

import numpy as np

from .utils import load_data
from .model import Model


def initialize_data(config, train=False, test=False):
    # print(os.path.abspath(config['data']['dataset_path']))
    print("Initializing data source...")
    train_source, test_source = load_data(**config['data'], cache=(train or test))
    if train:
        print("Loading training data...")
        train_source.load_all_data()
    if test:
        print("Loading test data...")
        test_source.load_all_data()
    
    print("Data initialization complete.")
    return train_source, test_source


def initialize_model(config, train_source, test_source):
    print("Initializing model...")
    data_config = config['data']
    model_config = config['model']
    model_param = deepcopy(model_config)
    model_param['train_source'] = train_source
    model_param['test_source'] = test_source
    model_param['train_pid_num'] = data_config['pid_num']
    batch_size = int(np.prod(model_config['batch_size']))
    model_param['save_name'] = '_'.join(map(str,[
        model_config['model_name'],
        data_config['dataset'],
        data_config['pid_num'],
        data_config['pid_shuffle'],
        model_config['hidden_dim'],
        model_config['margin'],
        batch_size,
        model_config['hard_or_full_trip'],
        model_config['frame_num'],
    ]))

    m = Model(**model_param)
    print("Model initialization complete.")
    return m, model_param['save_name']


def initialization(config, train=False, test=False):
    print("Initialzing...")
    WORK_PATH = config['WORK_PATH']
    os.chdir(WORK_PATH)
    os.environ["CUDA_VISIBLE_DEVICES"] = config["CUDA_VISIBLE_DEVICES"]
    train_source, test_source = initialize_data(config, train, test)
    return initialize_model(config, train_source, test_source)