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

import os
import struct
import numpy as np
import sys
result_root = sys.argv[1]
top1_acc = 0
top5_acc = 0
total = 0


for result_path in os.listdir(result_root):
    label = int(result_path.split('.')[0].split('_')[1])
    
    data_raw = np.fromfile(os.path.join(result_root, result_path), dtype=np.float16)
    sort_data = sorted(data_raw, reverse=True)
    data_raw = data_raw.tolist()
    result_top1 = data_raw.index(sort_data[0])
    result_top5 = []
    for i in range(5):
        result_top5.append(data_raw.index(sort_data[i]))

    total += 1
    top1_acc += 1 if label == result_top1 else 0
    top5_acc += 1 if label in result_top5 else 0
print('TOP1: ', top1_acc/total)
print('TOP5: ', top5_acc/total)

