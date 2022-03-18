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
import multiprocessing

max_bin=10
def preprocess(src_path, save_path, batch_size):
    files = os.listdir(src_path)

    output_data = [0]
    for i, file in enumerate(files):
        input_data = np.fromfile(os.path.join(src_path, file), dtype=np.float32)
        input_data = input_data.reshape(1, 3, 224, 224)
        
        if i % batch_size == 0:
            output_data = input_data
        else:
            output_data = np.concatenate((output_data, input_data), axis=0)

        # only save 10 bin files
        loop_id = (i + 1) // batch_size 
        if loop_id > max_bin:
            break 
        
        if (i + 1) % batch_size == 0:
            output_data.tofile("{}/img_{}_bs{}.bin".format(save_path, loop_id, batch_size))
            output_data = [0]


if __name__ == '__main__':
    if len(sys.argv) < 4:
        raise Exception("usage: python3 xxx.py [src_path] [save_path] [batch_size]")
    src_path = sys.argv[1]
    save_path = sys.argv[2]
    batch_size = int(sys.argv[3])
    src_path = os.path.realpath(src_path)
    save_path = os.path.realpath(save_path)

    if not os.path.isdir(save_path):
        os.makedirs(os.path.realpath(save_path))
    preprocess(src_path, save_path, batch_size)

