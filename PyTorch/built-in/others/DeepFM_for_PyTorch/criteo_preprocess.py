# -*- coding: utf-8 -*-

# Copyright 2020 Huawei Technologies Co., Ltd
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at# 
# 
#     http://www.apache.org/licenses/LICENSE-2.0# 
#     
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import configparser
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

print('Preprocessing data may take few minutes, please be patient...')

DATA_PATH = './train.txt'
DATA_INFO_PATH = './data.ini'

config = configparser.ConfigParser()

sparse_features = ['C' + str(i) for i in range(1, 27)]
dense_features = ['I' + str(i) for i in range(1, 14)]
target = ['label']
 
name_column = target + dense_features + sparse_features

print('1/4 Loading data...')
data = pd.read_csv(DATA_PATH, names=name_column, sep='\t')


print('2/4 Processing data...')
data[sparse_features] = data[sparse_features].fillna('-1', )
data[dense_features] = data[dense_features].fillna(0, )
for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])
mms = MinMaxScaler(feature_range=(0, 1))
data[dense_features] = mms.fit_transform(data[dense_features])
sparse_nunique = [str(data[feat].nunique()) for feat in sparse_features]
train, test = train_test_split(data, test_size=0.1, random_state=2020)

print('3/4 Writing processed data')
pd.DataFrame(train, columns=name_column).to_csv('./deepfm_trainval.txt', index=False, sep='\t')
pd.DataFrame(test, columns=name_column).to_csv('./deepfm_test.txt', index=False, sep='\t')


print('4/4 Writing data info')
config.add_section('data')
config.set('data', 'train_sample_num', str(train.shape[0]))
config.set('data', 'test_sample_num', str(test.shape[0]))
config.set('data', 'test_size', '0.1')
config.set('data', 'sparse_features', ",".join(sparse_features))
config.set('data', 'dense_features', ",".join(dense_features))
config.set('data', 'target', ",".join(target))
config.set('data', 'sparse_nunique', ",".join(sparse_nunique))

config.write(open(DATA_INFO_PATH, "w"))

