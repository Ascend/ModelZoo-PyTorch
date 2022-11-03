# -*- coding: utf-8 -*-

# Copyright 2022 Huawei Technologies Co., Ltd
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


from distutils.ccompiler import new_compiler
import os
import time
import random
import argparse
import configparser

import moxing as mox
import numpy as np
import pandas as pd

import torch
import torch.onnx

from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import DeepFM


parser = argparse.ArgumentParser(description='DeepFM for PyTorch')
parser.add_argument('--seed', default=1234, type=int,
                    help='seed for initializing training.')

parser.add_argument('--use_npu', default=False, action='store_true', help='8p distributed training')
parser.add_argument('--use_cuda', default=False, action='store_true', help='8p distributed training')
parser.add_argument('--device_id', default=0, type=int, help='device id')
parser.add_argument('--dist', default=False, action='store_true', help='8p distributed training')
parser.add_argument('--device_num', default=1, type=int, help='num of npu device for training')
parser.add_argument('--init_checkpoint', default='', type=str, help='init checkpoint for resume')

parser.add_argument('--amp', default=False, action='store_true',
                    help='use amp to train the model')
parser.add_argument('--loss-scale', default=1024, type=float,
                    help='loss scale using in amp, default -1 means dynamic')
parser.add_argument('--opt_level', default='O1', type=str,
                    help='apex opt level')

parser.add_argument('--obs_root', required=True, type=str, help='source file dir on obs')
parser.add_argument('--data_url', required=True, type=str, help='dataset repository path')
parser.add_argument('--train_url', required=True, type=str, help='model output path')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate for training')
parser.add_argument('--optim', default='adam', type=str, help='optimizer for model')
parser.add_argument('--test_size', default=0.1, type=float, help='data size for testing, while the rest for training')
parser.add_argument('--batch_size', default=1024, type=int, help='batch size for training and testing')
parser.add_argument('--epochs', default=3, type=int, help='epochs for training')
parser.add_argument('--steps', default=0, type=int, help='steps for training')


# data config
config = configparser.ConfigParser()


def fix_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    #copy obs files to environment
    local_root = os.getcwd()
    code_root = os.path.join(local_root, 'code')
    os.mkdir(code_root)
    dataset_root = os.path.join(local_root, 'dataset')
    os.mkdir(dataset_root)
    output_root = os.path.join(local_root, 'output')
    os.mkdir(output_root)

    mox.file.copy_parallel(args.obs_root, code_root)
    mox.file.copy_parallel(args.data_url, dataset_root)


    # data config
    config.read(dataset_root + '/data.ini')
    train_sample_num = int(config.get('data', 'train_sample_num'))
    test_sample_num = int(config.get('data', 'test_sample_num'))

    fix_random(args.seed)

    sparse_features = config.get('data', 'sparse_features').split(',')
    dense_features = config.get('data', 'dense_features').split(',')
    target = config.get('data', 'target').split(',')

    total_trainval_sample = train_sample_num
    nrows = total_trainval_sample // args.device_num
    skip_rows = list(range(1, 1 + args.device_id * nrows)) if args.device_num > 1 else None

    # 1.Loading preprocessed data, where label encoded for sparse features, 
    #   and simple Transformation for dense features is done
    print('Loading processed data...')
    start_time = time.time()
    data_trainval = pd.read_csv(dataset_root + '/deepfm_trainval.txt', sep='\t', skiprows=skip_rows, nrows=nrows)
    data_test = pd.read_csv(dataset_root + '/deepfm_test.txt', sep='\t')
    print('Data loaded in {}s'.format(time.time() - start_time))

    # 2.count #unique features for each sparse field,and record dense feature field name
    sparse_nunique_list = config.get('data', 'sparse_nunique').split(',')
    sparse_nunique = [int(sparse_nunique_list[i]) for i, _ in enumerate(sparse_nunique_list)]
    fixlen_feature_columns = [SparseFeat(feat, sparse_nunique[idx], embedding_dim=8)
                              for idx, feat in enumerate(sparse_features)] + [DenseFeat(feat, 1, )
                                                             for feat in dense_features]
    print(fixlen_feature_columns)

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)


    # 3.generate input data for model
    print('Generating input data for model...')
    start_time = time.time()
    train, test = data_trainval, data_test
    train_model_input = {name: train[name].astype(float) for name in feature_names}
    test_model_input = {name: test[name].astype(float) for name in feature_names}
    print('Input data generated in {}s'.format(time.time() - start_time))

    # 4.Define Model,train,predict and evaluate
    if args.dist:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29680'
        if args.use_npu:
            torch.distributed.init_process_group(backend='hccl', world_size=args.device_num, rank=args.device_id)
        elif args.use_cuda:
            torch.distributed.init_process_group(backend='nccl', world_size=args.device_num, rank=args.device_id)
        else:
            raise RuntimeError("Distributed training is not supported on this platfrom")
        print('distributed train enabled')

    device = 'cpu'
    if args.use_npu:
        device = 'npu:' + str(args.device_id)
        torch.npu.set_device(device)
    elif args.use_cuda:
        device = 'cuda:' + str(args.device_id)
        torch.cuda.set_device(device)
    print('train on: ', device)

    model = DeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
                   task='binary', dnn_hidden_units=(512, 256), dnn_dropout=0.5,
                   sparse_features_len=len(sparse_features),
                   device=device, l2_reg_linear=0, l2_reg_embedding=0, dist=args.dist)

    model.compile(args.optim, "binary_crossentropy",
                  metrics=["binary_crossentropy", "auc"], lr=args.lr)

    history = model.fit(train_model_input, train[target].values, batch_size=args.batch_size, epochs=args.epochs,
                        verbose=2, validation_split=0.1, args=args)
    pred_ans = model.predict(test_model_input, args.batch_size)

    Pth_root = os.path.join(local_root, 'deepfm-model.pth')
    onnx_root = os.path.join(local_root, 'deepfm-model.onnx')

    OBSPth_root = os.path.join(args.train_url, 'deepfm-model.pth')
    OBSonnx_root = os.path.join(args.train_url, 'deepfm-model.onnx')
    model.eval()

    print("export to pth...")
    torch.save(model,Pth_root)

    print("export to onnx....")
    a1 = torch.zeros(1).npu()
    a2 = torch.zeros(1,1).npu()
    arg = []
    for i in range(0, 26):
        arg.append(a1)
    for i in range(26,39):
        arg.append(a2)
    torch.onnx.export(model, arg, onnx_root, verbose=False, opset_version=11)

    print("export success")
    print("")
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
    print("saving pth and onnx to OBS ....")
    mox.file.copy(Pth_root, OBSPth_root)
    mox.file.copy(onnx_root, OBSonnx_root)

    print("succeed in saving pth to OBS.")

