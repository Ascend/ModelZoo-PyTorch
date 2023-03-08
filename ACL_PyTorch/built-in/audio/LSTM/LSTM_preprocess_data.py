# Copyright 2021 Huawei Technologies Co., Ltd
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

# !/usr/bin/python
# encoding=utf-8

import os
import sys
import copy
import time
import yaml
import shutil
import argparse
import numpy as np
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

sys.path.append('./')
from models.model_ctc import *
from utils.data_loader import Vocab, SpeechDataset, SpeechDataLoader

supported_rnn = {'nn.LSTM': nn.LSTM, 'nn.GRU': nn.GRU, 'nn.RNN': nn.RNN}
supported_activate = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid}

parser = argparse.ArgumentParser()
parser.add_argument('--conf', help='conf file for training')
parser.add_argument('--batchsize', help='batchsize for preprocessing')

class Config(object):
    batch_size = 4
    dropout = 0.1

def main():
    args = parser.parse_args()
    try:
        conf = yaml.safe_load(open(args.conf,'r'))
    except:
        print("Config file not exist!")
        sys.exit(1)

    opts = Config()
    for k,v in conf.items():
        setattr(opts, k, v)
        print('{:50}:{}'.format(k, v))

    # Data Loader
    batchsize = int(args.batchsize)
    vocab = Vocab(opts.vocab_file)
    dev_dataset = SpeechDataset(vocab, opts.test_scp_path, opts.test_lab_path, opts)
    dev_loader = SpeechDataLoader(dev_dataset, batch_size=batchsize, shuffle=False, num_workers=opts.num_workers,
                                  drop_last=True, pin_memory=True)

    bin_path = "./lstm_bin_bs" + args.batchsize
    if os.path.exists(bin_path):
        shutil.rmtree(bin_path)
    os.makedirs(bin_path)
    i = -1
    for data in dev_loader:
        i = i + 1
        print("[info] file", "===", i)
        inputs, input_sizes, targets, target_sizes, utt_list = data
        inputs_np = inputs.numpy()
        inputs_np.tofile(os.path.join(bin_path, "inputs_" + str(i) + '.bin'))

if __name__ == '__main__':
    main()
