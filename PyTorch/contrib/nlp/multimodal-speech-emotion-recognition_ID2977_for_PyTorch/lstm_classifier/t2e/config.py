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
import gensim
from create_vocab import Vocabulary, create_vocab

model_config = {
    #'gpu': 1,
    'npu': 0,
    '<PAD>': 0,
    '<SOS>': 1,
    '<EOS>': 2,
    '<UNK>': 3,
    'n_layers': 2,
    'dropout': 0.2,
    'output_dim': 6,  # number of classes
    'hidden_dim': 500,
    'n_epochs': 10000, #45000
    'batch_size': 128,  # carefully chosen
    'embedding_dim': 200,  # 50/100/200/300
    'bidirectional': True,
    'learning_rate': 0.0001,
    'model_code': 'bi_lstm_2_layer',
    'max_sequence_length': 20,
    'embeddings_dir': '/home/ma-user/modelarts/inputs/data_url_0/t2e/embeddings/'
}


from utils import generate_word_embeddings


def set_dynamic_hparams():
    try:

        with open('/home/ma-user/modelarts/inputs/data_url_0/t2e/vocab.pkl', 'rb') as f:
            vocab = pickle.load(f)
    except FileNotFoundError as e:
        vocab = create_vocab()
        generate_word_embeddings(vocab)

    model_config['vocab_size'] = vocab.size
    model_config['vocab_path'] = 'vocab.pkl'
    return model_config


model_config = set_dynamic_hparams()
