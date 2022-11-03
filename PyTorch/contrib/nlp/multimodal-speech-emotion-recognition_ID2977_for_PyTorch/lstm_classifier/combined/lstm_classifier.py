# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import sys
import pickle
import numpy as np
import torch
if torch.__version__ >= "1.8":
    import torch_npu
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from utils import load_data, evaluate, plot_confusion_matrix
import os
import time
from apex import amp
from config import model_config as config
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import itertools
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score

CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != CALCULATE_DEVICE:
    CALCULATE_DEVICE = f'npu:{CALCULATE_DEVICE}'
class LSTMClassifier(nn.Module):
    """docstring for LSTMClassifier"""
    def __init__(self, config):
        super(LSTMClassifier, self).__init__()
        self.n_layers = config['n_layers']
        self.dropout = config['dropout'] if self.n_layers > 1 else 0
        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.output_dim = config['output_dim']
        self.bidirectional = config['bidirectional']

        self.rnn = nn.LSTM(self.input_dim, self.hidden_dim, bias=True,
                           num_layers=self.n_layers, dropout=self.dropout,
                           bidirectional=self.bidirectional)
        self.out = nn.Linear(self.hidden_dim, self.output_dim)
        self.softmax = F.softmax

    def forward(self, input_seq):
        # input_seq =. [1, batch_size, input_size]
        rnn_output, (hidden, _) = self.rnn(input_seq)
        if self.bidirectional:  # sum outputs from the two directions
            rnn_output = rnn_output[:, :, :self.hidden_dim] +\
                        rnn_output[:, :, self.hidden_dim:]
        class_scores = F.softmax(self.out(rnn_output[0]), dim=1)
        return class_scores

if __name__ == '__main__':
    emotion_dict = {'ang': 0, 'hap': 1, 'sad': 2, 'fea': 3, 'sur': 4, 'neu': 5}

    torch.npu.set_device(CALCULATE_DEVICE)

    model = LSTMClassifier(config)
    model = model.to(CALCULATE_DEVICE)
    criterion = nn.CrossEntropyLoss()
    import apex
    optimizer = apex.optimizers.NpuFusedAdam(model.parameters(), lr=config['learning_rate'])
    model, optimizer = amp.initialize(model,optimizer,opt_level='O2',loss_scale=32.0,combine_grad=True)
    train_batches = load_data()
    test_pairs = load_data(test=True)

    best_acc = 0
    for epoch in range(config['n_epochs']):
        losses = []
        i = 0
        runtime = 0
        for batch in train_batches:
            start = time.time()
            inputs = batch[0].unsqueeze(0)  # frame in format as expected by model
            targets = batch[1]
            inputs = inputs.to(CALCULATE_DEVICE)
            targets = targets.to(CALCULATE_DEVICE)

            model.zero_grad()
            optimizer.zero_grad()

            predictions = model(inputs)
            predictions = predictions.to(CALCULATE_DEVICE)

            loss = criterion(predictions, targets)
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
            losses.append(loss.item())
            end = time.time()
            runtime += (end - start)
            i += 1
        if epoch % 1 == 0:
            print("epoch = " + str(epoch) + "; steptime = {:.3f}; loss = {:.3f}".format(runtime/i, loss.data))
        # evaluate
        with torch.no_grad():
            inputs = test_pairs[0].unsqueeze(0)
            targets = test_pairs[1]

            inputs = inputs.to(CALCULATE_DEVICE)
            targets = targets.to(CALCULATE_DEVICE)

            predictions = torch.argmax(model(inputs), dim=1)  # take argmax to get class id
            predictions = predictions.to(CALCULATE_DEVICE)

            # evaluate on cpu
            targets = np.array(targets.cpu())
            predictions = np.array(predictions.cpu())

            performance = evaluate(targets, predictions)
            if performance['acc'] > best_acc:
                print(performance)
                best_acc = performance['acc']
                # save model and results
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, './outputs/train_url_0/{}-combined-best_model.pth'.format(config['model_code']))

                with open('./outputs/train_url_0/{}-combined-best_performance.pkl'.format(config['model_code']), 'wb') as f:
                    pickle.dump(performance, f)
