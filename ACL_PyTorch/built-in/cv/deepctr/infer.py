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


import argparse
import aclruntime
import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
from torch.utils.data import DataLoader

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from deepctr_torch.inputs import SparseFeat, get_feature_names, \
    build_input_features
from ais_bench.infer.interface import InferSession


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default="./WDL.om", type=str,
                        help="which model to use:xDeepFM,WDL,AutoInt")
    parser.add_argument('--data_name_or_path', default="./movielens_sample.txt",
                        type=str, help='dir of data')
    parser.add_argument('--device_id', default=0, type=int,
                        help='which device to use')
    args = parser.parse_args()

    # fixed seed
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    data = pd.read_csv(args.data_name_or_path)
    sparse_features = ['movie_id', 'user_id', 'gender',
                       'age', 'occupation', 'zip']
    target = ['rating']

    # 1.Label Encoding for sparse features, and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    # 2.count unique features for each sparse field
    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                              for feat in sparse_features]
    linear_feature_columns = fixlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns +
                                      dnn_feature_columns)

    # 3.generate input data for model
    train, test = train_test_split(data, test_size=0.2)
    test_model_input = {name: test[name] for name in feature_names}
    feature_index = build_input_features(linear_feature_columns +
                                         dnn_feature_columns)

    x = []
    if isinstance(test_model_input, dict):
        x = [test_model_input[feature] for feature in feature_index]

    for i in range(len(x)):
        if len(x[i].shape) == 1:
            x[i] = np.expand_dims(x[i], axis=1)

    tensor_data = Data.TensorDataset(torch.from_numpy(np.concatenate(x,
                                                                     axis=-1)))

    test_loader = DataLoader(dataset=tensor_data, shuffle=False, batch_size=256)

    # 4.infer
    pred_ans = []
    model = InferSession(args.device_id, args.model_path)
    for _, x_test in enumerate(test_loader):
        x = x_test[0].float()

        y_pred = model.infer([x])[0]
        pred_ans.append(y_pred)

    result = np.concatenate(pred_ans).astype('float64')

    print('test MSE', round(mean_squared_error(test[target].values, result), 4))
