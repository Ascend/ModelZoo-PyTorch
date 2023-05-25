# Copyright 2023 Huawei Technologies Co., Ltd
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

import random
import numpy as np
from blocklm_utils import ConstructBlockStrategy
from argparse import Namespace


# rng = random.Random()
# span_lengths = [2, 3, 4, 2, 3, 4]
# length = 100
#
# counts = np.array([0] * length)
# for _ in range(10000):
#     rng.shuffle(span_lengths)
#     spans = ConstructBlockStrategy.sample_spans(span_lengths, length, rng)
#     for start, end in spans:
#         counts[start: end] += 1
# print(counts)
def main():
    args = Namespace()
    args.seq_length = 10
    args.eod_token = 0

    strategy = ConstructBlockStrategy(args, None, bert_ratio=0.4, max_seq_length=128)
    counts = np.array([0] * 10)
    for _ in range(10000):
        spans = strategy.sample_span_in_document(np.array([1, 2, 3, 0, 4, 5, 6, 7, 9, 0], dtype=np.long), [1, 1],
                                                 random.Random())
        for start, end in spans:
            counts[start: end] += 1

    print(counts)
