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
import torch
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from deepctr_torch.inputs import SparseFeat, get_feature_names
from deepctr_torch.models import WDL, AutoInt, xDeepFM


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default="WDL", type=str,
                        help="which model to use:xDeepFM,WDL,AutoInt")
    parser.add_argument('--data_name_or_path', default="./movielens_sample.txt",
                        type=str, help='dir of data')
    parser.add_argument('--device_id', default='0', type=str,
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

    # 1.Label Encoding for sparse features,
    # and do simple Transformation for dense features
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
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # 4.Define Model,train,predict and evaluate
    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:' + args.device_id

    model = None
    if args.model_name == 'WDL':
        model = WDL(linear_feature_columns, dnn_feature_columns,
                    task='regression', device=device)
    elif args.model_name == 'AutoInt':
        model = AutoInt(linear_feature_columns, dnn_feature_columns,
                        task='regression', device=device)
    elif args.model_name == 'xDeepFM':
        model = xDeepFM(linear_feature_columns, dnn_feature_columns,
                        task='regression', device=device)
    else:
        print('please enter the correct name of model')
    model.compile('adam', 'mse', metrics=['mse'], )

    history = model.fit(train_model_input, train[target].values,
                        batch_size=256, epochs=1000, verbose=2,
                        validation_split=0.2)
    torch.save(model.state_dict(), args.model_name+'_weight.h5')
    print('#########test')
    checkpoint = torch.load(args.model_name+'_weight.h5', map_location='cpu')
    model.load_state_dict(checkpoint)
    print('#########checkpoint created')
    pred_ans = model.predict(test_model_input, batch_size=256)
    print('#########test')
    print('test MSE', round(mean_squared_error(test[target].values,
                                               pred_ans), 4))
