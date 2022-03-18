# -*- coding: utf-8 -*-
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

import argparse
import os
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import *

parser = argparse.ArgumentParser(description="Pytorch DCN Training")
parser.add_argument('--npu_id', default=0, type=int, help='npu device id for training')
parser.add_argument('--dist', default=False, action='store_true', help='8p distributed training')
parser.add_argument('--device_num', default=1, type=int, help='num of npu device for training')
parser.add_argument('--trainval_path', required=True, type=str, help='train and validation dataset path')
parser.add_argument('--test_path', required=True, type=str, help='test dataset path')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate for training')
parser.add_argument('--use_fp16', default=True, action='store_true', help='use fp16 training')
parser.add_argument('--early_stop_step', default=-1, type=int,
                    help='mannual setting of training step, -1 for no early stop')


TOTAL_TRAIN_VAL_SAMPLE = int(45840616 * 0.93)
TOTAL_TEST_SAMPLE = int(45840616 * 0.07)
if __name__ == "__main__":
    args = parser.parse_args()

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]
    target = ['label']

    nrows = TOTAL_TRAIN_VAL_SAMPLE // args.device_num
    if args.device_num > 1:
        skip_rows = list(range(1, 1 + args.npu_id * nrows))
    else:
        skip_rows = None
    data_trainval = pd.read_csv(args.trainval_path, sep='\t', skiprows=skip_rows, nrows=nrows)
    data_test = pd.read_csv(args.test_path, sep='\t')

    # 2.count #unique features for each sparse field,and record dense feature field name

    SPARSE_NUNIQUE = [1460, 583, 10131227, 2202608, 305, 24, 12517, 633, 3, 93145, 5683, 8351593, 3194, 27, 14992, 5461306, 10, 5652, 2173, 4, 7046547, 18, 15, 286181, 105, 142572]
    fixlen_feature_columns = [SparseFeat(feat, SPARSE_NUNIQUE[idx], embedding_dim=int(SPARSE_NUNIQUE[idx]**0.08)*8)
                              for idx, feat in enumerate(sparse_features)] + [DenseFeat(feat, 1, )
                                                              for feat in dense_features]
    print(fixlen_feature_columns)

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model
    train, test = data_trainval, data_test

    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # 4.Define Model,train,predict and evaluate
    if args.dist:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29680'
        torch.distributed.init_process_group(backend='hccl', world_size=args.device_num, rank=args.npu_id)
        print('=============>distributed train')

    device = 'npu:' + str(args.npu_id)
    torch.npu.set_device(device)
    print('======================>train on', device)

    model = DCN(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
                   task='binary', device=device, dnn_use_bn=True, dnn_hidden_units=(1024, 1024, 1024), cross_num=6,
                l2_reg_linear=0, l2_reg_embedding=0, l2_reg_cross=0, l2_reg_dnn=0, dist=args.dist)

    model.compile("adam", "binary_crossentropy",
                  metrics=["binary_crossentropy", "auc"], )

    model.fit(train_model_input, train[target].values, batch_size=1024, lr=args.lr, use_fp16=args.use_fp16, epochs=2, verbose=2,
                        validation_split=0.07, early_stop_step=args.early_stop_step)
    if args.early_stop_step < 0:
        pred_ans = model.predict(test_model_input, 1024)
        print("")
        print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
        print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
