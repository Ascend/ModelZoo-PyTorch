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
import cv2
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
    filelist = os.listdir(result_dir)
    objcnt = len(filelist)
    for idx in range(len(filelist)):
        if idx == objcnt:
            break
        file_dir = os.path.join(result_dir, "{}_1.bin".format(idx))
        data = np.fromfile(file_dir, dtype='float32')
        results.append(data)
    results = np.concatenate(results)
    results = results.reshape(objcnt, cls_cnts)
        
    # read labels
    fr=open(label_dir, 'rb')
    labels_file = pickle.load(fr)
    labels = labels_file[1]
    labels = labels[:objcnt] # the shortcut of labels
    print('Top 1: {:.2f}%'.format(100 * topk_accuracy(results, labels, 1)))
    print('Top 5: {:.2f}%'.format(100 * topk_accuracy(results, labels, 5)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='kinetics-skeleton postprocess')
    parser.add_argument('--result_dir', 
        default='./result/dumpOutput_device1/', help='data file path')
    parser.add_argument('--label_path', 
        default='./data/kinetics-skeleton/val_label.pkl', 
        help='label file path')
    args = parser.parse_args()
    # create input_data
    st_gcn_postprocess(args.result_dir, args.label_path)
