# Copyright 2023 Huawei Technologies Co., Ltd
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
import stat
import torch
import numpy as np
from sklearn.metrics import classification_report

def rename(path):
    '''
    outputs of ais_bench end with "_0.txt",
    it would cause outputs files can not match label files,
    rename and remove "_0" could fix it. 
    '''
    file_names = os.listdir(path)
    for file_name in file_names:
        if not file_name.endswith('.txt'):
            continue
        idx = file_name.split('_')[0]
        src_path = os.path.join(path, file_name)
        dist_path = os.path.join(path, idx + '.txt')
        os.rename(src_path, dist_path)

def load_output_to_results(path):
    file_names = os.listdir(path)
    results = []
    for file_name in file_names:
        if not file_name.endswith('.txt'):
            continue
        file_name = os.path.join(path, file_name)
        with open(file_name) as f:
            for l in f:
                data = list(map(float, l.strip().split()))
                pred = torch.tensor(data).argmax(axis=-1).cpu().numpy()
                results.append(int(pred))
    
    return results

def load_labels(path):
    file_names = os.listdir(path)
    labels = []
    for file_name in file_names:
        file_name = os.path.join(path, file_name)
        with open(file_name) as f:
            for l in f:
                data = l.strip()
                labels.append(int(data))
    
    return labels

if __name__ == '__main__':
    output_path, label_path = sys.argv[1], sys.argv[2]
    rename(output_path)
    preds = load_output_to_results(output_path)
    true_labels = load_labels(label_path)
    eval_result = classification_report(true_labels,
                                        preds,
                                        digits=4)
    print(f'Results: {eval_result}')
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL 
    modes = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open('result.txt', flags, modes), 'w') as main_f:
        main_f.write(eval_result)
