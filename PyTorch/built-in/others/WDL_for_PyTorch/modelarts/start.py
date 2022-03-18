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

import torch

from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import WDL

# ---------modelarts modification-----------------
import glob
import moxing as mox
from pthtar2onnx import convert

CACHE_TRAINING_URL = '/cache/training'
CACHE_DATA_URL = '/cache/data_url/'
# ---------modelarts modification end-------------


def args_parser():
    parser = argparse.ArgumentParser(description='Wide&Deep')
    # ------------modelarts modification-----------------
    parser.add_argument('--data_url', metavar='DIR', default='/cache/data_url', help='path to dataset')
    parser.add_argument('--train_url', default="/cache/training", type=str, help="setting dir of training output")
    parser.add_argument('--onnx', default=True, help="convert pth model to onnx")
    # -----------modelarts modification end-------------

    parser.add_argument('--seed', default=1234, type=int,
                        help='seed for initializing training.')
    parser.add_argument('--device_id', default=0, type=int, help='device id')
    parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument('--dist', default=False, action='store_true', help='8p distributed training')
    parser.add_argument('--device_num', default=1, type=int,
                        help='num of npu device for training')
    parser.add_argument('--amp', default=True, action='store_true',
                        help='use amp to train the model')
    parser.add_argument('--loss_scale', default=1024, type=float,
                        help='loss scale using in amp, default -1 means dynamic')
    parser.add_argument('--opt_level', default='O1', type=str,
                        help='apex opt level')
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


def convert_pth_to_onnx():
    pth_pattern = os.path.join(CACHE_TRAINING_URL, 'checkpoint.pth.tar')
    pth_file_list = glob.glob(pth_pattern)
    if not pth_file_list:
        print(f"can't find pth {pth_pattern}")
        return
    pth_file = pth_file_list[0]
    onnx_path = pth_file.split(".")[0] + '.onnx'
    convert(pth_file, onnx_path)


if __name__ == "__main__":
    args = args_parser()
    print(args)

    fix_random(args.seed)

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]
    target = ['label']

    # count #unique features for each sparse field,and record dense feature field name
    start_time = time.time()

    # ---------------modelarts modification-----------------
    if not os.path.exists(CACHE_DATA_URL):
        os.makedirs(CACHE_DATA_URL)
    mox.file.copy_parallel(args.data_url, CACHE_DATA_URL)
    print("training data finish copy to %s." % CACHE_DATA_URL)
    # --------------modelarts modification end--------------

    data_trainval = pd.read_pickle(os.path.join(CACHE_DATA_URL, 'wdl_trainval.pkl'))
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
    print('Input data generated in {}s'.format(time.time() - start_time))

    # Define Model,train,predict and evaluate
    args.device_num = torch.npu.device_count()
    if args.dist:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29680'

        args.rank = args.device_id
        torch.distributed.init_process_group(backend='hccl', world_size=args.device_num, rank=args.rank)
        print('distributed train enabled')

    device = 'npu:' + str(args.device_id)
    torch.npu.set_device(device)
    print('train on: ', device)

    # --------------modelarts modification---------
    if not os.path.exists(CACHE_TRAINING_URL):
        os.makedirs(CACHE_TRAINING_URL, 0o755)
    args.checkpoint_save_path = CACHE_TRAINING_URL
    # --------------modelarts modification end-----

    model = WDL(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
                task='binary', dnn_hidden_units=(512, 256, 128), dnn_dropout=0.5, device=device, l2_reg_linear=1e-4,
                l2_reg_embedding=1e-4, dist=args.dist)

    model.compile('adam', 'binary_crossentropy',
                  metrics=['binary_crossentropy', 'auc'], lr=args.lr, args=args)

    history = model.fit(train_model_input, data_trainval[target].values, batch_size=args.batch_size, epochs=args.epochs,
                        verbose=2,
                        validation_split=0.3, args=args)

    if args.onnx:
        if args.amp and args.opt_level == 'O1':
            from apex import amp
            with amp.disable_casts():
                convert_pth_to_onnx()
        else:
            convert_pth_to_onnx()

    # --------------modelarts modification--------------------
    mox.file.copy_parallel(CACHE_TRAINING_URL, args.train_url)
    # --------------modelarts modification end----------------
