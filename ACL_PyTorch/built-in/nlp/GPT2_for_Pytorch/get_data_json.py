
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
import json

def prepare_data(data_path, save_dir):

    data = []
    pre_path = os.listdir(data_path)
    for mid_path in pre_path:
        path_ = os.path.join(data_path,mid_path)
        re_path = os.listdir(path_)
        for pp in re_path:
            p_ = os.path.join(path_,pp)
            with open(p_, 'r', encoding='utf8') as f:
                lines = f.readlines()
                for line in lines:
                    data.append(json.loads(line)['text'])
            break
    with open(save_dir, 'w') as f:
        json.dump(data,f)
        

if __name__ == '__main__':
    data_path = sys.argv[1]
    save_dir = sys.argv[2]
    prepare_data(data_path, save_dir)


