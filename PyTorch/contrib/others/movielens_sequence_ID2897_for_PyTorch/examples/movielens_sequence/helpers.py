# Copyright 2018 The Cornac Authors. All Rights Reserved.
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
# ============================================================================
# Copyright 2022 Huawei Technologies Co., Ltd
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
import pandas as pd

from tabulate import tabulate


def _load_data(filename, columns=None):

    data = pd.read_json(filename, lines=True)
    data = data.sort_values('validation_mrr', ascending=False)

    mrr_cols = ['validation_mrr', 'test_mrr']

    if columns is None:
        columns = [x for x in data.columns if
                   (x not in mrr_cols and x != 'hash')]

    cols = data.columns
    cols = mrr_cols + columns

    return data[cols]


def _print_df(df):

    print(tabulate(df, headers=df.columns,
                   showindex=False,
                   tablefmt='pipe'))


def print_data():

    cnn_data = _load_data('results/cnn_results.txt',
                          ['residual',
                           'nonlinearity',
                           'loss',
                           'num_layers',
                           'kernel_width',
                           'dilation',
                           'embedding_dim'])
    _print_df(cnn_data[:5])

    lstm_data = _load_data('results/lstm_results.txt')

    _print_df(lstm_data[:5])

    pooling_data = _load_data('results/pooling_results.txt')

    _print_df(pooling_data[:5])


if __name__ == '__main__':
    print_data()
