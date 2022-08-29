# Copyright 2022 Huawei Technologies Co., Ltd
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
import argparse
import numpy as np
import pickle


def topk_accuracy(score, label, k=1):
    rank = score.argsort()
    hit_top_k = [l in rank[i, -k:] for i, l in enumerate(label)]
    accuracy =sum(hit_top_k) * 1.0 / len(hit_top_k)
    return accuracy


def st_gcn_postprocess(result_dir, label_dir):
    results = []
    labels = []
    cls_cnts = 400
    # read data
    if not os.path.isdir(result_dir):
        print("the result file path error:", result_dir)
    file_list = os.listdir(result_dir)
    file_list.sort(key=lambda x:int(x[:-4]))
    for file_name in file_list:
        file_dir = os.path.realpath(os.path.join(
        args.result_dir, file_name))
        data = []
        with open(file_dir, "r") as f:
            for line in f:
                data.append(float(line.strip('\n')))
        data_np = np.array(data)
        results.append(data_np)
    results = np.concatenate(results)
    results = results.reshape(len(file_list), cls_cnts)

    ranks = results.argsort()
    infer_result = []
    for rank in ranks:
        infer_result.append(rank[-5:])
    print("infer_results ",infer_result)
    file = open('infer_result.txt', 'w')
    file.write(str(infer_result))
    file.close()
        
    # read labels
    fr=open(label_dir, 'rb')
    labels_file = pickle.load(fr)
    labels = labels_file[1]
    print('Top 1: {:.2f}%'.format(100 * topk_accuracy(results, labels, 1)))
    print('Top 5: {:.2f}%'.format(100 * topk_accuracy(results, labels, 5)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='kinetics-skeleton postprocess')
    parser.add_argument('--result_dir', 
        default='../data/results', help='data file path')
    parser.add_argument('--label_dir', 
        default='../data/kinetics-skeleton/val_label.pkl', 
        help='label file path')
    args = parser.parse_args()
    # create input_data
    st_gcn_postprocess(args.result_dir, args.label_dir)