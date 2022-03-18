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
import sys
import torch
import numpy as np

from DeepRL.deep_rl.utils.misc import *


if __name__ == '__main__':
    input_file = sys.argv[1]
    bin_file = sys.argv[2]
    out_file = sys.argv[3]
    files = os.listdir(input_file)
    mkdir(bin_file)
    mkdir(out_file)
    for file in files:
        state = torch.load(input_file+'/'+file)
        mystate = np.array(state).astype(np.float32)
        mystate.tofile(os.path.join(bin_file, file.split('.')[0] + ".bin"))
