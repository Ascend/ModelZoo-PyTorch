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
import pickle
import numpy as np
import pandas as pd
from config import model_config as config

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

import itertools
import matplotlib.pyplot as plt


def load_data(batched=True, test=False, file_dir='./inputs/data_url_0/combined/combined_features.pkl'):
    bs = config['batch_size']
    ftype = 'test' if test else 'train'

    with open('{}'.format(file_dir), 'rb') as f:
        features = pickle.load(f)

    x = features['x_{}'.format(ftype)]
    y = features['y_{}'.format(ftype)]
    data = (x, y)
    if test or not batched:
        return [torch.FloatTensor(data[0]), torch.LongTensor(data[1])]
    data = list(zip(data[0], data[1]))
    n_iters = len(data) // bs
    batches = []
    for i in range(1, n_iters + 1):
        input_batch = []
        output_batch = []
        for e in data[bs * (i-1):bs * i]:
            input_batch.append(e[0])
            output_batch.append(e[1])
        batches.append([torch.FloatTensor(input_batch),
                        torch.LongTensor(output_batch)])
    return batches


def evaluate(targets, predictions):
    performance = {
        'acc': accuracy_score(targets, predictions),
        'f1': f1_score(targets, predictions, average='macro'),
        'precision': precision_score(targets, predictions, average='macro'),
        'recall': recall_score(targets, predictions, average='macro')}
    return performance


def plot_confusion_matrix(targets, predictions, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # plt.figure(figsize=(8,8))
    cm = confusion_matrix(targets, predictions)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
