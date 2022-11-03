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
from lstm_classifier import LSTMClassifier
from config import model_config as config
from utils import load_data


# Load test data
test_pairs = load_data(test=True)
inputs, lengths, targets = test_pairs

# Load pretrained model
model = LSTMClassifier(config)
checkpoint = torch.load('./outputs/train_url_0/{}-best_model.pth'.format(config['model_code']),
                        map_location='cpu')
model.load_state_dict(checkpoint['model'])

with torch.no_grad():
    # Predict
    predict_probas = model(inputs, lengths).cpu().numpy()

    with open('./outputs/train_url_0/text_lstm_classifier.pkl', 'wb') as f: #../../pred_probas/text_lstm_classifier.pkl
        pickle.dump(predict_probas, f)
