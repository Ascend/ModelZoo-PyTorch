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
model_config = {
    'gpu': 1,
    'n_layers': 2,
    'dropout': 0.2,
    'output_dim': 6,  # number of classes
    'hidden_dim': 256,
    'input_dim': 2472,
    'batch_size': 200,  # carefully chosen
    'n_epochs': 20000, #55000
    'learning_rate': 0.001,
    'bidirectional': True,
    'model_code': 'bi_lstm'
}
