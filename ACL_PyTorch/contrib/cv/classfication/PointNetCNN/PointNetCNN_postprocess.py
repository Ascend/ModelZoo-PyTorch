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
import numpy as np
import sys
import os

def postprocess(data_label_path,data_om_path):
    label_nums = 2468 # 标签数量
    acc=0
    label_path = data_label_path
    my_om_path = os.listdir(data_om_path)
    om_path = '{}/{}/data'.format(data_om_path,my_om_path[0])
    for i in range(1, label_nums+1):
        data_label_path = '{}{}.npy'.format(label_path,str(i))
        data_om_path = '{}{}_0.txt'.format(om_path,str(i))
        data_label = np.load(data_label_path)
        data_om = np.loadtxt(data_om_path,dtype=np.float32)
        if data_label[0][0]==np.argmax(data_om):
            acc+=1

    print(acc*1.0/label_nums)

if __name__ == '__main__':
    data_label_path = sys.argv[1]
    data_om_path = sys.argv[2]
    postprocess(data_label_path,data_om_path)
