# -*- coding: utf-8 -*-
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
# @Time    : 2021-04-20 10:36
# @Author  : WenYi
# @Contact : 1244058349@qq.com
# @Description :  script description


import torch
if torch.__version__ >= "1.8":
    import torch_npu
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')


# data process
def data_preparation():
    # The column names are from
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
                    'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
                    'income_50k']
    
    # Load the dataset in Pandas
    train_df = pd.read_csv(
        './data/adult.data',
        delimiter=',',
        header=None,
        index_col=None,
        names=column_names
    )
    other_df = pd.read_csv(
        './data/adult.test',
        delimiter=',',
        header=None,
        index_col=None,
        names=column_names
    )
    
    train_df['tag'] = 1
    other_df['tag'] = 0
    other_df.dropna(inplace=True)
    other_df['income_50k'] = other_df['income_50k'].apply(lambda x: x[:-1])
    data = pd.concat([train_df, other_df])
    data.dropna(inplace=True)
    # First group of tasks according to the paper
    label_columns = ['income_50k', 'marital_status']
    
    # categorical columns
    categorical_columns = ['workclass', 'education', 'occupation', 'relationship', 'race', 'sex', 'native_country']
    for col in label_columns:
        if col == 'income_50k':
            data[col] = data[col].apply(lambda x: 0 if x == ' <=50K' else 1)
        else:
            data[col] = data[col].apply(lambda x: 0 if x == ' Never-married' else 1)
            
    # feature engine
    for col in column_names:
        if col not in label_columns + ['tag']:
            if col in categorical_columns:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])
            else:
                mm = MinMaxScaler()
                data[col] = mm.fit_transform(data[[col]]).reshape(-1)
    data = data[['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'occupation',
                 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
                 'income_50k', 'marital_status', 'tag']]
    
    # user feature, item feature
    user_feature_dict, item_feature_dict = dict(), dict()
    for idx, col in enumerate(data.columns):
        if col not in label_columns + ['tag']:
            if idx < 7:
                if col in categorical_columns:
                    user_feature_dict[col] = (len(data[col].unique())+1, idx)
                else:
                    user_feature_dict[col] = (1, idx)
            else:
                if col in categorical_columns:
                    item_feature_dict[col] = (len(data[col].unique())+1, idx)
                else:
                    item_feature_dict[col] = (1, idx)
    
    # Split the other dataset into 1:1 validation to test according to the paper
    train_data, test_data = data[data['tag'] == 1], data[data['tag'] == 0]
    train_data.drop('tag', axis=1, inplace=True)
    test_data.drop('tag', axis=1, inplace=True)
    
    # val data
    # train_data, val_data = train_test_split(train_data, test_size=0.5, random_state=2021)
    return train_data, test_data, user_feature_dict, item_feature_dict


class TrainDataSet(Dataset):
    def __init__(self, data):
        self.feature = data[0]
        self.label1 = data[1]
        self.label2 = data[2]
        
    def __getitem__(self, index):
        feature = self.feature[index]
        label1 = self.label1[index]
        label2 = self.label2[index]
        return feature, label1, label2
        
    def __len__(self):
        return len(self.feature)
