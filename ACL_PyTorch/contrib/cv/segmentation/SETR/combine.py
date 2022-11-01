# Copyright 2022 Huawei Technologies Co., Ltd
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
import shutil

old_path =sys.argv[1]
filenames = os.listdir(old_path)
print(filenames)
target_path = r"./new_input_bin"
    
for file in filenames:
    sonDir = r"./input_bin/" + file
    print(sonDir)
    for root, dirs, files in os.walk(sonDir):
        if len(files) > 0:
            for f in files:
                newDir = sonDir + '/' + f
                shutil.move(newDir, target_path)
        else:
            print(sonDir + "dir is exists")