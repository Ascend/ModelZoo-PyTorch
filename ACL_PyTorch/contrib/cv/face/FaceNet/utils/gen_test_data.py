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
import numpy as np


def gen_bin(save_name, data_shape):
    data_bin = np.random.random(data_shape)
    data_bin.tofile(os.path.join(save_name))


if __name__ == '__main__':
    save_path = sys.argv[1]
    shape = sys.argv[2]
    shape = list(map(int, shape.split(',')))
    gen_bin(save_path, shape)
