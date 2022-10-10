# Copyright 2020 Huawei Technologies Co., Ltd
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
import json
import torch
import sys


result_dir = sys.argv[1]  # result_dir：推理得到的输出文件夹
label_dir = sys.argv[2]  # label_dir：标注文件的路径
json_dir = sys.argv[3]
result_dir = os.listdir(result_dir)

# 处理annotation文件,得到一个label字典，key为类名称，value为类的索引
# label = {'v_Skiing_g04_c03': '80', 'v_SoccerPenalty_g02_c04': '84', ......}
label = dict()
f = open(label_dir)
x = f.readlines()
f.close()
for i in range(len(x)):
    class_name = x[i].split(' ')[0].split('/')[1]
    class_idx = x[i].split(' ')[2].replace('\n', '').replace('\r', '')
    label[class_name] = class_idx

file_name = result_dir

num_correct_top1 = 0
num_total = len(file_name)
num_other_file = 0
# 统计top1正确的个数
for file_idx in range(num_total):
    x = file_name[file_idx]
    if 'sumary' in x : 
        num_other_file+=1
        continue
    f = open(os.path.join(sys.argv[1], x))
    scores = f.readlines()
    s = [[],[],[],[],[],[],[],[],[],[]]
    for i, score in enumerate(scores):
        score_ = score.replace('\n', '').replace('\r', '').split(' ')  # score：list[str]
        #print(score_)
        s[i] = [float(i) for i in score_ if i!='']
    #score = score[0:1010]
    #score = [float(i) for i in score]
    f.close()
    #s = [[], [], [], [], [], [], [], [], [], []]
    #for i in range(10):
    #    s[i] = score[101*i:101*i + 101]  # 对于score中的1010个分数，每隔101个将其取出放到s数组中
    cls_score = torch.tensor(s).mean(dim=0)  # 对10个clips得到的输出结果求平均
    max_value = cls_score[0]
    idx = 0
    for i in range(len(cls_score)):
        if cls_score[i] >= max_value:
            max_value = cls_score[i]
            idx = i
    if label[x.split('.')[0].replace('_0', '')] == str(idx):
        num_correct_top1 += 1

top1_acc = num_correct_top1/(num_total-num_other_file)
result_dict = {"top1_acc": top1_acc}
print(result_dict)
json_str = json.dumps(result_dict)
with open(json_dir, 'w') as json_file:
    json_file.write(json_str)
