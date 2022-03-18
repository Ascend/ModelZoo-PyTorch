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

import numpy as np
import os
import json
import torch
import sys


def get_file(root):
    catfile = os.path.join(root, 'synsetoffset2category.txt')
    cat = {}
    with open(catfile, 'r') as f:
        for line in f:
            ls = line.strip().split()
            cat[ls[0]] = ls[1]

    id2cat = {v: k for k, v in cat.items()}
    split = 'test'
    meta = {}
    splitfile = os.path.join(root, 'train_test_split', 'shuffled_{}_file_list.json'.format(split))
    filelist = json.load(open(splitfile, 'r'))

    for item in cat:
        meta[item] = []

    for file in filelist:
        _, category, uuid = file.split('/')
        if category in cat.values():
            meta[id2cat[category]].append((os.path.join(root, category, 'points', uuid + '.pts'),
                                           os.path.join(root, category, 'points_label', uuid + '.seg')))

    datapath = []
    for item in cat:
        for fn in meta[item]:
            datapath.append((item, fn[0], fn[1]))

    classes = dict(zip(sorted(cat), range(len(cat))))
    f = open('name2label.txt', 'w')
    for index in range(len(datapath)):
        fn = datapath[index]
        cls = classes[datapath[index][0]]
        f.write(fn[1] + " ")
        f.write(str(cls) + "\n")
        f.flush()
    f.close()

    return datapath


def preprocess_bs1(datapath, save_path):
    npoints = 2500
    total = len(datapath)
    for index in range(total):
        fn = datapath[index]
        point_set = np.loadtxt(fn[1]).astype(np.float32)
        seg = np.loadtxt(fn[2]).astype(np.int64)

        choice = np.random.choice(len(seg), npoints, replace=True)
        point_set = point_set[choice, :]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist

        file_name = fn[1].split('/')[-1]
        point_set = point_set.transpose(1, 0)
        point_set.tofile(os.path.join(save_path, file_name.split('.')[0] + ".bin"))


def preprocess_bs16(datapath, save_path):
    npoints = 2500
    total = len(datapath)
    iter_total = total // 16
    for k in range(iter_total + 1):
        start = k * 16
        if k == iter_total:
            end = total
        else:
            end = k * 16 + 16
        batch_data = None
        for index in range(start, end):
            fn = datapath[index]
            point_set = np.loadtxt(fn[1]).astype(np.float32)
            seg = np.loadtxt(fn[2]).astype(np.int64)

            choice = np.random.choice(len(seg), npoints, replace=True)

            point_set = point_set[choice, :]

            point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
            dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
            point_set = point_set / dist
            point_set = point_set.transpose(1, 0)
            if index == start:
                batch_data = point_set
            else:
                batch_data = np.concatenate((batch_data, point_set), axis=0)
        if k < 10:
            file_name = "bin_batch00" + str(k)
        elif k >= 10 and k < 100:
            file_name = "bin_batch0" + str(k)
        else:
            file_name = "bin_batch" + str(k)
        batch_data.tofile(os.path.join(save_path, file_name + ".bin"))


if __name__ == '__main__':
    root = sys.argv[1]
    save_path = sys.argv[2]
    batch_size = sys.argv[3]
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    datapath = get_file(root)
    if batch_size.endswith("16"):
        preprocess_bs16(datapath, save_path)
    else:
        preprocess_bs1(datapath, save_path)
