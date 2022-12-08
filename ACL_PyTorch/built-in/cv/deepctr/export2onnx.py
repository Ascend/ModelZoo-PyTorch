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
import os
import torch
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from deepctr_torch.inputs import SparseFeat, get_feature_names
from deepctr_torch.models import WDL, AutoInt, xDeepFM

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default="WDL", type=str, help="which model to use:xDeepFM,WDL,AutoInt")
    parser.add_argument('--data_name_or_path', default="./movielens_sample.txt", type=str, help='dir of data')
    args = parser.parse_args()

    # fixed seed
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    data = pd.read_csv(args.data_name_or_path)
    sparse_features = ['movie_id', 'user_id', 'gender', 'age', 'occupation', 'zip']
    target = ['rating']

    # 1.Label Encoding for sparse features, and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    # 2.count unique features for each sparse field
    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique()) for feat in sparse_features]
    linear_feature_columns = fixlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    model = None
    if args.model_name == 'WDL':
        model = WDL(linear_feature_columns, dnn_feature_columns, task='regression')
    elif args.model_name == 'AutoInt':
        model = AutoInt(linear_feature_columns, dnn_feature_columns, task='regression')
    elif args.model_name == 'xDeepFM':
        model = xDeepFM(linear_feature_columns, dnn_feature_columns, task='regression')
    else:
        print('please enter the correct name of model')

    checkpoint = torch.load(args.model_name+'_weight.h5', map_location='cpu')
    model.load_state_dict(checkpoint)

    model.eval()
    input_name = ['input']
    output_name = ['output']
    input_data = torch.zeros(40, 6)

    output = args.model_name + '.onnx'
    output = os.path.join('./model', output)
    torch.onnx.export(model, (input_data), output, input_names=input_name, output_names=output_name,
                      opset_version=13, export_params=True, verbose=True, do_constant_folding=True)
    print('success')
