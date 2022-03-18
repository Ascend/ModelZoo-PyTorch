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
path = os.getcwd()

if not os.path.exists(os.path.join(path,'1')):
    os.makedirs(os.path.join(path,'1'))
with open('FDDB_dets.txt', 'r') as f:
    
    while(True):
        img_name = f.readline().strip('\n').replace('/', '_')
        if img_name:
            pass
        else:
            break
        
        raw = f.readline().strip('\n').split('.')[0]
        file_name = ''.join([img_name, '.txt'])
        
        os.chdir(os.path.join(path, '1'))
        with open(file_name, 'w') as new_file:
            new_file.write(img_name+'\n')
            new_file.write(raw+'\n')
            for i in range(int(raw)):
                new_file.write(f.readline())
        os.chdir(path)
