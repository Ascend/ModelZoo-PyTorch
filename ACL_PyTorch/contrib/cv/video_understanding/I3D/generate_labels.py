# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import csv

def convert_label(s, keep_whitespaces=False):
    if not keep_whitespaces:
        return s.replace('"', '').replace(' ', '_')
    else:
        return s.replace('"', '')


def line_to_map(x):
    video = f'{x[1]}'
    label = class_mapping[convert_label(x[0])]
    return video, label


train_file = 'annotations/kinetics_train.csv'
val_file = 'annotations/kinetics_val.csv'

csv_reader = csv.reader(open(train_file))
# skip the first line
next(csv_reader)

labels_sorted = sorted(set([convert_label(row[0]) for row in csv_reader]))
class_mapping = {label: str(i) for i, label in enumerate(labels_sorted)}

csv_reader = csv.reader(open(val_file))
next(csv_reader)
val_list = [line_to_map(x) for x in csv_reader]

with open('kinetics400_label.txt','w') as f:
    for i in val_list:
        if i == val_list[-1]:
            f.write(' '.join(i))
        else:
            f.write(' '.join(i)+'\n')

path = os.listdir('rawframes_val')
print('Total number of videos:',len(path))

with open('kinetics400_label.txt','r') as f:
    data = f.readlines()

print('File kinetics400_label.txt successfully generated.')

dic = dict()

for d in data:
    temp = d.replace('\n','').split()
    dic[temp[0]] = temp[1]
res = []
for i in path:
    if i == path[-1]:
        path_temp = os.listdir('rawframes_val/{}'.format(i))
        total_rawframes = len(path_temp)
        res.append(' '.join([i, str(total_rawframes), dic[i]]))
    else:
        path_temp = os.listdir('rawframes_val/{}'.format(i))
        total_rawframes = len(path_temp)
        res.append(' '.join([i,str(total_rawframes),dic[i]])+'\n')

with open('kinetics400_val_list_rawframes.txt','w') as f:
    f.writelines(res)

print('File kinetics400_val_list_rawframes.txt successfully generated.')