# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
import pickle

import numpy as np

from config import pickle_file, sos_id, eos_id

print('loading {}...'.format(pickle_file))
with open(pickle_file, 'rb') as file:
    data = pickle.load(file)
VOCAB = data['VOCAB']
IVOCAB = data['IVOCAB']

print('loading bigram_freq.pkl...')
with open('bigram_freq.pkl', 'rb') as file:
    bigram_freq = pickle.load(file)

OUT_LIST = ['<sos>比赛很快便城像一边到的局面第二规合<eos>', '<sos>比赛很快便呈像一边到的局面第二规合<eos>', '<sos>比赛很快便城向一边到的局面第二规合<eos>',
            '<sos>比赛很快便呈向一边到的局面第二规合<eos>', '<sos>比赛很快便城像一边到的局面第二回合<eos>']
GT = '比赛很快便呈向一边倒的局面第二回合<eos>'

print('calculating prob...')
prob_list = []
for out in OUT_LIST:
    print(out)
    out = out.replace('<sos>', '').replace('<eos>', '')
    out = [sos_id] + [VOCAB[ch] for ch in out] + [eos_id]
    prob = 1.0
    for i in range(1, len(out)):
        prob *= bigram_freq[(out[i - 1], out[i])]
    prob_list.append(prob)

prob_list = np.array(prob_list)
prob_list = prob_list / np.sum(prob_list)
print(prob_list)
