# BSD 3-Clause License

# Copyright (c) Soumith Chintala 2016,
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
SPARSE_FEATURES_NUM = 27
DENSE_FEATURES_NUM = 14

sparse_features = ['C' + str(i) for i in range(1, SPARSE_FEATURES_NUM)]
dense_features = ['I' + str(i) for i in range(1, DENSE_FEATURES_NUM)]
target = ['label']

name_column = target + dense_features + sparse_features

csv_path = os.path.abspath(sys.argv[1])
data = pd.read_csv(csv_path, names=name_column, sep='\t')
data[sparse_features] = data[sparse_features].fillna('-1', )
data[dense_features] = data[dense_features].fillna(0, )

for feat in sparse_features:
    print(feat)
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])
mms = MinMaxScaler(feature_range=(0, 1))
data[dense_features] = mms.fit_transform(data[dense_features])
train, test = train_test_split(data, test_size=0.07, random_state=2020)
pd.DataFrame(train, columns=name_column).to_csv(os.path.dirname(csv_path) + '/train_after_preprocess_trainval_0.93.txt', index=False, sep='\t')
pd.DataFrame(test, columns=name_column).to_csv(os.path.dirname(csv_path) + '/train_after_preprocess_test_0.07.txt', index=False, sep='\t')