# Copyright 2021 Huawei Technologies Co., Ltd
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
import argparse
import os
import json
import numpy as np


def read_info_from_json(json_path):
    
    if os.path.exists(json_path) is False:
        print(json_path, 'is not exist')
    with open(json_path, 'r') as f:
        load_data = json.load(f)
        file_info = load_data['filesinfo']
        return file_info


def postProcesss(result_path):
    file_info = read_info_from_json(result_path)
    outputs = []
    for i in file_info.items():        
        res_path = i[1]['outfiles'][0]
        #print(res_path)
        ndata = np.loadtxt(res_path)
        outputs.append(ndata)
    return outputs

def cre_groundtruth_dict_fromtxt():
    """
    :param filename: file contains the imagename and label number
    :return: dictionary key imagename, value is label number
    """
    gtfile_path = './val_label.txt'
    labels=[]
    with open(gtfile_path, 'r')as f:
        for line in f.readlines():
            temp = line.strip().split(" ")
            labels.append(int(temp[1]))
    return labels

def top_k_accuracy(scores, labels, topk=(1, )):
    res = []
    labels = np.array(labels)[:, np.newaxis]
    for k in topk:
        max_k_preds = np.argsort(scores[:50000], axis=1)[:, -k:][:, ::-1]
        #print(max_k_preds)
        #print(len(max_k_preds))
        match_array = np.logical_or.reduce(max_k_preds == labels, axis=1)
        topk_acc_score = match_array.sum() / match_array.shape[0]
        res.append(topk_acc_score)
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='postprocess of googlenet')
    parser.add_argument('--result_path')
    opt = parser.parse_args()
    outputs = postProcesss(opt.result_path)
    labels = cre_groundtruth_dict_fromtxt()
    print('Evaluating top_k_accuracy ...')
    top_acc = top_k_accuracy(outputs, labels, topk=(1, 5))
    print(f'\ntop{1}_acc\t{top_acc[0]:.4f}')
    print(f'\ntop{5}_acc\t{top_acc[1]:.4f}')
