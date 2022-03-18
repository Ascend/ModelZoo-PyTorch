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
import shutil
path = os.getcwd()  # D:\PythonProject\FDDB_Evaluation
pre_dir = os.path.join(path, '1')  # D:\PythonProject\FDDB_Evaluation\1
# D:\PythonProject\FDDB_Evaluation\pred_sample
cur_path = os.path.join(path, 'pred_sample')
if not os.path.exists(cur_path):
        os.makedirs(cur_path)
        
for dir_name in os.listdir(cur_path):
    print(dir_name)
    # D:\PythonProject\FDDB_Evaluation\pred_sample\1
    tmp_path = os.path.join(cur_path, dir_name.strip('\n'))
    print(tmp_path)
    for data in os.listdir(tmp_path):
        pre_file = os.path.join(pre_dir, data)
        cur_file = os.path.join(tmp_path, data)
        shutil.move(pre_file, cur_file)
