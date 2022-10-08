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

import json
import os  
import sys
def file_name(file_dir):   
    L=[]
    s = 0
    for dirpath, dirnames, filenames in os.walk(file_dir):  
        for file in filenames:  
            if os.path.splitext(file)[1] == '.json':  
                with open(os.path.join(dirpath, file), 'r') as f:
                    result = json.load(f)
                    s = s+result["throughput"]
    return s 

s = file_name("result")
print("====310P performance data====")
print('310P bs{} fps:{}'.format(sys.argv[1], s/5))