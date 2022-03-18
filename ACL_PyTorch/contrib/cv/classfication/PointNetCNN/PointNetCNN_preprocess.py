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

import torch
import numpy as np
import os
import sys
import provider
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

TEST_FILES = provider.getDataFiles(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))
BATCH_SIZE = 1

def preprocess(save_path,label_save_path):
    i = 0
    test_file_idxs = np.arange(0, len(TEST_FILES))
    for fn in range(len(TEST_FILES)):
        current_data, current_label = provider.loadDataFile(TEST_FILES[test_file_idxs[fn]])
        current_data = current_data[:, 0:1024, :]
        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE
        for batch_idx in range(num_batches):
            i += 1
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx + 1) * BATCH_SIZE
            label = current_label[start_idx:end_idx]
            rotated_data = provider.rotate_point_cloud(current_data[start_idx:end_idx, :, :])
            jittered_data = provider.jitter_point_cloud(rotated_data)  # P_Sampled
            P_sampled = np.array(torch.from_numpy(jittered_data).float(), dtype=np.float32)
            
            P_sampled.tofile(os.path.join(save_path, "data" +str(i) + ".bin"))
            np.save(os.path.join(label_save_path,'label'+str(i)),label)

if __name__ == "__main__":
    save_path = sys.argv[1]
    label_save_path = sys.argv[2]
    save_path = os.path.realpath(save_path)
    if not os.path.isdir(save_path):
        os.makedirs(os.path.realpath(save_path))
    if not os.path.isdir(label_save_path):
        os.makedirs(os.path.realpath(label_save_path))
    preprocess( save_path,label_save_path)
