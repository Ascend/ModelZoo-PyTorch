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
# ============================================================================

import os
import pickle as plk
import numpy as np

from tqdm import tqdm

def get_lengths(mode, feature_name):
    fd = data[mode][feature_name]
    max_len = fd.shape[1]
    
    c_sum = np.sum(fd, axis=-1)
    lengths = []
    for i in tqdm(range(fd.shape[0])):
        null = True
        zeros = np.zeros([fd.shape[1], fd.shape[2]])
        cur_length = max_len
        for j in range(max_len):
            if c_sum[i][j] == 0:
                cur_length = j
                null = False
                break
        if cur_length == 0:
            cur_length = 1
        lengths.append(cur_length)
    return lengths

with open('/home/sharing/disk3/dataset/multimodal-sentiment-dataset/ALL/mosei/unaligned_50.pkl', 'rb') as lf:
    data = plk.load(lf)

def handleData(mode):
    # data[mode]['audio_lengths'], _ = get_lengths(mode, 'feature_A')
    # data[mode]['vision_lengths'], _ = get_lengths(mode, 'feature_V')
    data[mode]['audio_lengths'] = get_lengths(mode, 'audio')
    data[mode]['vision_lengths'] = get_lengths(mode, 'vision')

handleData('train')
handleData('valid')
handleData('test')

with open('/home/sharing/disk3/dataset/multimodal-sentiment-dataset/ALL/mosei/unaligned_50.pkl', 'wb') as df:
    plk.dump(data, df, protocol = 4)