#!/usr/bin/env python
# coding=utf-8
"""
 Copyright(C) 2022. Huawei Technologies Co.,Ltd. All rights reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import os
import random
import pickle
import numpy as np
from scipy.io import loadmat

np.random.seed(0)
random.seed(0)

def generate_data_description(save_path):
    """
    create a dataset description file, which consists of images, labels
    """
    dataset = dict()
    dataset['description'] = 'peta'
    dataset['root'] = './images/'
    dataset['test_image'] = []
    dataset['att'] = []
    dataset['att_name'] = []
    dataset['selected_attribute'] = range(35)
    # load PETA.MAT
    petaPath = './PETA.mat'
    data = loadmat(petaPath)
    attri_num = 105
    sample_num = 19000
    for idx in range(attri_num):
        dataset['att_name'].append(data['peta'][0][0][1][idx, 0][0])

    for idx in range(sample_num):
        dataset['test_image'].append('%05d.png' % (idx + 1))
        dataset['att'].append(data['peta'][0][0][0][idx, 4:].tolist())
    with open(os.path.join(save_path, 'peta_dataset.pkl'), 'wb') as f:
        pickle.dump(dataset, f)


def create_trainvaltest_split(file_name):
    """
    create a dataset split file, which consists of index of the train/val/test splits
    """
    partition = dict()
    partition['trainval'] = []
    partition['train'] = []
    partition['val'] = []
    partition['test'] = []
    partition['weight_trainval'] = []
    partition['weight_train'] = []
    # load PETA.MAT
    petaPath = './PETA.mat'
    data = loadmat(petaPath)
    for idx in range(5):
        train = (data['peta'][0][0][3][idx][0][0][0][0][:, 0] - 1).tolist()
        val = (data['peta'][0][0][3][idx][0][0][0][1][:, 0] - 1).tolist()
        test = (data['peta'][0][0][3][idx][0][0][0][2][:, 0] - 1).tolist()
        trainval = train + val
        partition['train'].append(train)
        partition['val'].append(val)
        partition['trainval'].append(trainval)
        partition['test'].append(test)
        # weight
        weight_trainval = np.mean(data['peta'][0][0][0][trainval, 4:].astype('float32') == 1, axis=0).tolist()
        weight_train = np.mean(data['peta'][0][0][0][train, 4:].astype('float32') == 1, axis=0).tolist()
        partition['weight_trainval'].append(weight_trainval)
        partition['weight_train'].append(weight_train)
    with open(file_name, 'wb') as f:
        pickle.dump(partition, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="peta dataset")
    parser.add_argument(
        '--save_dir',
        type=str,
        default = './')
    parser.add_argument(
        '--traintest_split_file',
        type=str,
        default = "./peta_partition.pkl")
    args = parser.parse_args()
    save_dir = args.save_dir
    traintest_split_file = args.traintest_split_file

    generate_data_description(save_dir)
    create_trainvaltest_split(traintest_split_file)
