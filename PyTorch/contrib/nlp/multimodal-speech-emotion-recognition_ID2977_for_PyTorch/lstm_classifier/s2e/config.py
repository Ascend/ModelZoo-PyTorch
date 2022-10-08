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
    'gpu': 0,
    'bidirectional': False,
    'input_dim': 8,
    'hidden_dim': 50,
    'output_dim': 6,  # number of classes
    'dropout': 0.2,
    'learning_rate': 0.01,
    'batch_size': 1567,  # carefully chosen
    'n_epochs': 20000, #55000
    'n_layers': 2,
    'model_code': 'basic_lstm'
}
