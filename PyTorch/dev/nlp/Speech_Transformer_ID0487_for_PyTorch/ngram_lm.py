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
import collections
import pickle

import nltk
import numpy as np
from tqdm import tqdm

from config import pickle_file

with open(pickle_file, 'rb') as file:
    data = pickle.load(file)
char_list = data['IVOCAB']
vocab_size = len(char_list)
samples = data['train']
bigram_counter = collections.Counter()

for sample in tqdm(samples):
    text = sample['trn']
    # text = [char_list[idx] for idx in text]
    tokens = list(text)
    bigrm = nltk.bigrams(tokens)
    # print(*map(' '.join, bigrm), sep=', ')

    # get the frequency of each bigram in our corpus
    bigram_counter.update(bigrm)

# what are the ten most popular ngrams in this Spanish corpus?
print(bigram_counter.most_common(10))

temp_dict = dict()
for key, value in bigram_counter.items():
    temp_dict[key] = value

print('smoothing and freq -> prob')
bigram_freq = dict()
for i in tqdm(range(vocab_size)):
    freq_list = []
    for j in range(vocab_size):
        if (i, j) in temp_dict:
            freq_list.append(temp_dict[(i, j)])
        else:
            freq_list.append(1)

    freq_list = np.array(freq_list)
    freq_list = freq_list / np.sum(freq_list)

    assert (len(freq_list) == vocab_size)
    bigram_freq[i] = freq_list

print(len(bigram_freq[0]))
with open('bigram_freq.pkl', 'wb') as file:
    pickle.dump(bigram_freq, file)
