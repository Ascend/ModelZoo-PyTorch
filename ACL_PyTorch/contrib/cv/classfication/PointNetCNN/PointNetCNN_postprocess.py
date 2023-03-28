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


def postprocess(data_label_path, data_om_path):
    label_nums = len(os.listdir(data_om_path))
    acc = 0
    for i in range(1, label_nums + 1):
        data_label = '{}{}.npy'.format(data_label_path, str(i))
        data_path = '{}/data{}_0.txt'.format(data_om_path, str(i))
        label = np.load(data_label)
        data_om = np.loadtxt(data_path, dtype=np.float32)
        if label[0][0] == np.argmax(data_om):
            acc += 1

    print(acc / label_nums)


if __name__ == '__main__':
    label_path = sys.argv[1]
    om_path = sys.argv[2]
    postprocess(label_path, om_path)
