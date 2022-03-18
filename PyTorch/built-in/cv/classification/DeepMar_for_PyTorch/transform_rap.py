# Copyright [yyyy] [name of copyright owner]
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
import numpy as np
import random
import cPickle as pickle
from scipy.io import loadmat

np.random.seed(0)
random.seed(0)

def make_dir(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

def generate_data_description(save_dir):
    """
    create a dataset description file, which consists of images, labels
    """
    dataset = dict()
    dataset['description'] = 'rap'
    dataset['root'] = './dataset/rap/RAP_dataset/'
    dataset['image'] = []
    dataset['att'] = []
    dataset['att_name'] = []
    dataset['selected_attribute'] = range(51)
    # load Rap_annotation.mat
    data = loadmat(open('./dataset/rap/RAP_annotation/RAP_annotation.mat', 'r'))
    for idx in range(51):
        dataset['att_name'].append(data['RAP_annotation'][0][0][6][idx][0][0])
    
    for idx in range(41585):
        dataset['image'].append(data['RAP_annotation'][0][0][5][idx][0][0])
        dataset['att'].append(data['RAP_annotation'][0][0][1][idx, :].tolist())

    with open(os.path.join(save_dir, 'rap_dataset.pkl'), 'w+') as f:
        pickle.dump(dataset, f)

def create_trainvaltest_split(traintest_split_file):
    """
    create a dataset split file, which consists of index of the train/val/test splits
    """
    partition = dict()
    partition['trainval'] = []
    partition['test'] = []
    partition['weight_trainval'] = []
    # load RAP_annotation.mat
    data = loadmat(open('./dataset/rap/RAP_annotation/RAP_annotation.mat', 'r'))
    for idx in range(5):
        trainval = (data['RAP_annotation'][0][0][0][idx][0][0][0][0][0,:]-1).tolist()
        test = (data['RAP_annotation'][0][0][0][idx][0][0][0][1][0,:]-1).tolist()
        partition['trainval'].append(trainval)
        partition['test'].append(test)
        # weight
        weight_trainval = np.mean(data['RAP_annotation'][0][0][1][trainval, :].astype('float32')==1, axis=0).tolist()
        partition['weight_trainval'].append(weight_trainval)
    with open(traintest_split_file, 'w+') as f:
        pickle.dump(partition, f)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="rap dataset")
    parser.add_argument(
        '--save_dir',
        type=str,
        default='./dataset/rap/')
    parser.add_argument(
        '--traintest_split_file',
        type=str,
        default="./dataset/rap/rap_partition.pkl")
    args = parser.parse_args()
    save_dir = args.save_dir
    traintest_split_file = args.traintest_split_file

    generate_data_description(save_dir)
    create_trainvaltest_split(traintest_split_file)
