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
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from config import model_config as config
from utils import load_data, evaluate, load_word_embeddings, plot_confusion_matrix

import time
from apex import amp
import pandas as pd
import numpy as np
import pickle
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


CALCULATE_DEVICE = 'npu:0'

class LSTMClassifier(nn.Module):
    """docstring for LSTMClassifier"""
    def __init__(self, config):
        super(LSTMClassifier, self).__init__()
        self.dropout = config['dropout']
        self.n_layers = config['n_layers']
        self.hidden_dim = config['hidden_dim']
        self.output_dim = config['output_dim']
        self.vocab_size = config['vocab_size']
        self.embedding_dim = config['embedding_dim']
        self.bidirectional = config['bidirectional']

        self.embedding = nn.Embedding.from_pretrained(
            load_word_embeddings(), freeze=False)

        self.rnn = nn.LSTM(self.embedding_dim, self.hidden_dim, bias=True,
                           num_layers=self.n_layers, dropout=self.dropout,
                           bidirectional=self.bidirectional)
        self.n_directions = 2 if self.bidirectional else 1
        self.out = nn.Linear(self.n_directions * self.hidden_dim, self.output_dim)
        self.softmax = F.softmax

    def forward(self, input_seq, input_lengths):
        max_seq_len, bs = input_seq.size()
        # input_seq =. [max_seq_len, batch_size]
        embedded = self.embedding(input_seq)

        rnn_output, (hidden, _) = self.rnn(embedded)
        rnn_output = torch.cat((rnn_output[-1, :, :self.hidden_dim],
                                rnn_output[0, :, self.hidden_dim:]), dim=1)
        # sum hidden states
        class_scores = F.softmax(self.out(rnn_output), dim=1)
        return class_scores


if __name__ == '__main__':
    emotion_dict = {'ang': 0, 'hap': 1, 'sad': 2, 'fea': 3, 'sur': 4, 'neu': 5}

    # device = 'cuda:{}'.format(config['gpu']) if \
    #          torch.cuda.is_available() else 'cpu'

    if 'npu' in CALCULATE_DEVICE:
        torch.npu.set_device(CALCULATE_DEVICE)

    model = LSTMClassifier(config)
    model = model.to(CALCULATE_DEVICE)
    criterion = nn.NLLLoss()
    import apex
    optimizer = apex.optimizers.NpuFusedAdam(model.parameters(), lr=config['learning_rate'])
    model, optimizer = amp.initialize(model,optimizer,opt_level='O2',loss_scale=32.0,combine_grad=True)

    train_batches = load_data()
    test_batch = load_data(test=True)

    best_acc = 0


    for epoch in range(config['n_epochs']):

        losses = []
        i = 0
        runtime, sum_runtime = 0, 0
        for batch in train_batches:
            start = time.time()
            inputs, input_lengths, targets = batch
            inputs = inputs.to(CALCULATE_DEVICE)
            input_lengths = input_lengths.to(CALCULATE_DEVICE)
            targets = targets.to(CALCULATE_DEVICE)

            model.zero_grad()
            optimizer.zero_grad()

            predictions = model(inputs, input_lengths)
            predictions = predictions.to(CALCULATE_DEVICE)


            loss = criterion(predictions, targets)
            with amp.scale_loss(loss,optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
            losses.append(loss.item())


            end = time.time()
            runtime += (end - start)
            i += 1
        if epoch % 100 == 0 :  #否则输出太多了
            print("epoch = " + str(epoch) + "; steptime = {:.3f}; loss = {:.3f}".format(runtime/i, loss.data))
        ##sum_runtime += runtime/i
        # evaluate
        with torch.no_grad():
            inputs, lengths, targets = test_batch

            inputs = inputs.to(CALCULATE_DEVICE)
            lengths = lengths.to(CALCULATE_DEVICE)
            targets = targets.to(CALCULATE_DEVICE)

            predictions = torch.argmax(model(inputs, lengths), dim=1)  # take argmax to get class id
            predictions = predictions.to(CALCULATE_DEVICE)

            # evaluate on cpu
            targets = np.array(targets.cpu())
            predictions = np.array(predictions.cpu())

            # Get results
            # plot_confusion_matrix(targets, predictions,
            #                       classes=emotion_dict.keys())
            performance = evaluate(targets, predictions)
            if performance['acc'] > best_acc:
                best_acc = performance['acc']
                print(performance)
                # save model and results
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, '/home/ma-user/modelarts/outputs/train_url_0/{}-t2e-best_model.pth'.format(config['model_code']))

                with open('/home/ma-user/modelarts/outputs/train_url_0/{}-t2e-best_performance.pkl'.format(
                        config['model_code']), 'wb') as f:
                    pickle.dump(performance, f)


