#Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import os
PATH_ = os.path.dirname(os.path.abspath(__file__))

f = open(PATH_+"/label.txt", 'r')
lines = f.readlines()
isFirst = True
labels = []
count = 0
imgs_path = []
words = []
for line in lines:
    line = line.rstrip()
    if line.startswith('#'):
        if isFirst is True:
            isFirst = False
        else:
            labels_copy = labels.copy()
            words.append(labels_copy)
            labels.clear()
        path = line[2:]+"\n"
        imgs_path.append(path)
    else:
        line = line.split(' ')
        label = [float(x) for x in line]
        labels.append(label)
    words.append(labels)
fout = open("wider_val.txt", "w")
fout.writelines(imgs_path)
