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

import os
import numpy as np
import sys

'''
sys.argv[1]: om_output
sys.argv[2]: ground_truth
'''
om_output_files = sorted(os.listdir(sys.argv[1]))

output_labels = []
# 读取om输出
for file in om_output_files:
    with open(sys.argv[1] + file, mode='r') as f:
        content = f.read().split(' ')[:-1]
        content = list(map(lambda x: float(x), content))
        content = np.array(content)
        output_labels.append(np.argmax(content))

# 读取ground_truth
with open(sys.argv[2], mode='r') as f:
    ground_truth = list(map(lambda x: int(x.rstrip('\n').split(' ')[1]), f.readlines()))

count = 0
for i in range(len(output_labels)):
    if ground_truth[i] == output_labels[i]:
        count += 1
        
print(f"accuracy: {count / len(output_labels)}")
# print(count, len(output_labels))
# print(output_labels)
