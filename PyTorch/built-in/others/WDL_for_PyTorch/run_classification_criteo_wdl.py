# -*- coding: utf-8 -*-

# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# less required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


import os
import time
import random
import argparse

import numpy as np
import pandas as pd

import sklearn
import torch

from sklearn.metrics import log_loss, roc_auc_score

from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import WDL


def args_parser():
    parser = argparse.ArgumentParser(description='Wide&Deep')
    parser.add_argument('--seed', default=1234, type=int,
                        help='seed for initializing training.')
    parser.add_argument('--device_id', default=0, type=int, help='device id')
    parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument('--dist', default=False, action='store_true', help='8p distributed training')
    parser.add_argument('--device_num', default=1, type=int,
                        help='num of npu device for training')
    parser.add_argument('--amp', default=False, action='store_true',
                        help='use amp to train the model')
    parser.add_argument('--loss_scale', default=1024, type=float,
                        help='loss scale using in amp, default -1 means dynamic')
    parser.add_argument('--opt_level', default='O1', type=str,
                        help='apex opt level')
    parser.add_argument('--data_path', required=True, type=str, help='train data, and is to be')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--checkpoint_save_path', default='./', type=str, metavar='PATH',
                        help='path to save latest checkpoint')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate for training')
    parser.add_argument('--batch_size', default=1024, type=int, help='batch size for training')
    parser.add_argument('--eval_batch_size', default=16000, type=int, help='batch size for testing')
    parser.add_argument('--epochs', default=3, type=int, help='epochs for training')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='record of the start epoch to run')
    parser.add_argument('--sparse_embed_dim', default=4, type=int, help='The embedding dims for sparse features')
    parser.add_argument('--steps', default=0, type=int, help='steps for training')

    parser_args, _ = parser.parse_known_args()
    return parser_args


def fix_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    args = args_parser()
    print(args)

    fix_random(args.seed)

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]
    target = ['label']

    # count #unique features for each sparse field,and record dense feature field name
    start_time = time.time()

    data_trainval = pd.read_pickle(os.path.join(args.data_path, 'wdl_trainval.pkl'))
    data_test = pd.read_pickle(os.path.join(args.data_path, 'wdl_test.pkl'))

    print('Data loaded in {}s'.format(time.time() - start_time))

    sparse_nunique = [1460, 583, 10131227, 2202608, 305, 24, 12517, 633, 3, 93145, 5683, 8351593, 3194, 27, 14992,
                      5461306, 10, 5652, 2173, 4, 7046547, 18, 15, 286181, 105, 142572]
    fixlen_feature_columns = [SparseFeat(feat, sparse_nunique[idx], embedding_dim=args.sparse_embed_dim)
                              for idx, feat in enumerate(sparse_features)] + [DenseFeat(feat, 1, )
                                                                              for feat in dense_features]
    print(fixlen_feature_columns)

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)

    # generate input data for model
    print('Generating input data for model...')
    start_time = time.time()
    train_model_input = {name: data_trainval[name].astype(float) for name in feature_names}
    test_model_input = {name: data_test[name].astype(float) for name in feature_names}
    print('Input data generated in {}s'.format(time.time() - start_time))

    # Define Model,train,predict and evaluate
    args.device_num = int(os.environ["RANK_SIZE"])
    if args.dist:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29680'

        args.rank = args.device_id
        torch.distributed.init_process_group(backend='hccl', world_size=args.device_num, rank=args.rank)
        print('distributed train enabled')

    device = 'npu:' + str(args.device_id)
    torch.npu.set_device(device)
    print('train on: ', device)

    model = WDL(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
                task='binary', dnn_hidden_units=(512, 256, 128), dnn_dropout=0.5, device=device, l2_reg_linear=1e-4,
                l2_reg_embedding=1e-4, dist=args.dist)

    model.compile("adam", "binary_crossentropy",
                  metrics=["binary_crossentropy", "auc"], lr=args.lr, args=args)

    history = model.fit(train_model_input, data_trainval[target].values, batch_size=args.batch_size, epochs=args.epochs,
                        verbose=2,
                        validation_split=0.3, args=args)

    pred_ans = model.predict(test_model_input, args.eval_batch_size)
    print("test LogLoss", round(log_loss(data_test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(data_test[target].values, pred_ans), 4))
