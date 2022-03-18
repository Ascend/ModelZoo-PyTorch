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
correct, total = 0, 0

for result_path in os.listdir(result_root):
    label = int(result_path.split('.')[0].split('_')[1])
    
    data_raw = np.fromfile(os.path.join(result_root, result_path), dtype=np.float16)
    result = int(np.argmax(data_raw))
    total += 1
    correct += 1 if label == result else 0
print('acc: ', correct/total)
