"""
Copyright 2020 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


import os
import json
import numpy as np
import scipy.io as sio
import argparse


parser = argparse.ArgumentParser(description='Output', add_help=False)
parser.add_argument('--output', metavar='DIR',
                    help='output file')
parser.add_argument('--input-dir', metavar='DIR',
                    help='input directory')


def eval_parser():
    file = sio.loadmat('meta.mat')
    fr = open("ILSVRC2012_validation_ground_truth.txt", "r")
    eval_dir = []
    line = fr.readline()
    while line:
        idx = int(line) - 1
        dir_name = file["synsets"][idx][0][1][0]
        eval_dir.append(dir_name)
        line = fr.readline()
    fr.close()
    return eval_dir


def result2cls(input_dir):
    root = input_dir
    total = np.zeros(50000,dtype='int')
    for i in range(50000):
        line = "ILSVRC2012_val_000" + str(i+1).zfill(5) + "_0.txt"
        file = open(root+line, "r")
        res = file.readline().split(' ')[:-1]
        file.close()
        a = np.array(res, dtype=float)
        b = a.argmax()
        total[i] = b
    return total


def cls2dir(res_cls):
    file = sio.loadmat('meta.mat')
    lst = []
    count = [0]*1000
    for i in range(1000):
        idx = i
        dir_name = file["synsets"][idx][0][1][0]
        lst.append([dir_name])
    lst.sort()
    res_dir=[]
    for i in res_cls:
        dir_name=lst[i][0]
        res_dir.append(dir_name)
    return res_dir


def get_result(eval_dir, res_dir):
    count = 0
    for i in range(50000):
        if eval_dir[i] == res_dir[i]:
            count = count + 1
    return count/500


def main(args):
    eval_dir = eval_parser()
    res_cls = result2cls(args.input_dir)
    res_dir = cls2dir(res_cls)
    acc1 = get_result(eval_dir, res_dir)
    res_dict = {"title":"Overall statistical evaluation",
                "value":[{"key":"Number of images", "value":"50000"},
                         {"key":"Number of classes", "value":"1000"},
                         {"key":"Top1 accuracy", "value":acc1}]}
    with open(args.output, "w") as f:
        json.dump(res_dict, f)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
