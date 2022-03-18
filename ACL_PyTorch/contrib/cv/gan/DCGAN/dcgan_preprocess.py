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

"""
This script randomly generates 'noise' data following normal distribution as the input of model.
"""

import os
import sys
import numpy as np
import torch


def preprocess(save_path):
    total_noise = 8192
    shape = (100, 1, 1)
    for i in range(total_noise):
        print("generate noise %04d..." % i)
        noise = torch.randn(shape, dtype=torch.float)
        noise = np.array(noise).astype(np.float32)
        noise.tofile(os.path.join(save_path, "noise_%04d.bin" % i))


if __name__ == "__main__":
    save_path = sys.argv[1]
    save_path = os.path.realpath(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    preprocess(save_path)
