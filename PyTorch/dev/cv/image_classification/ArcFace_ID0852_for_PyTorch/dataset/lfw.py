#!/usr/bin/env python
# encoding: utf-8
#
# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================
#
'''
@author: wujiyang
@contact: wujiyang@hust.edu.cn
@file: lfw.py.py
@time: 2018/12/22 10:00
@desc: lfw dataset loader
'''

import numpy as np
import cv2
import os
import torch.utils.data as data

import torch
import torchvision.transforms as transforms

def img_loader(path):
    try:
        with open(path, 'rb') as f:
            img = cv2.imread(path)
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            return img
    except IOError:
        print('Cannot load image ' + path)

class LFW(data.Dataset):
    def __init__(self, root, file_list, transform=None, loader=img_loader):

        self.root = root
        self.file_list = file_list
        self.transform = transform
        self.loader = loader
        self.nameLs = []
        self.nameRs = []
        self.folds = []
        self.flags = []

        with open(file_list) as f:
            pairs = f.read().splitlines()[1:]
        for i, p in enumerate(pairs):
            p = p.split('\t')
            if len(p) == 3:
                nameL = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
                nameR = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[2]))
                fold = i // 600
                flag = 1
            elif len(p) == 4:
                nameL = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
                nameR = p[2] + '/' + p[2] + '_' + '{:04}.jpg'.format(int(p[3]))
                fold = i // 600
                flag = -1
            self.nameLs.append(nameL)
            self.nameRs.append(nameR)
            self.folds.append(fold)
            self.flags.append(flag)

    def __getitem__(self, index):

        img_l = self.loader(os.path.join(self.root, self.nameLs[index]))
        img_r = self.loader(os.path.join(self.root, self.nameRs[index]))
        imglist = [img_l, cv2.flip(img_l, 1), img_r, cv2.flip(img_r, 1)]

        if self.transform is not None:
            for i in range(len(imglist)):
                imglist[i] = self.transform(imglist[i])

            imgs = imglist
            return imgs
        else:
            imgs = [torch.from_numpy(i) for i in imglist]
            return imgs

    def __len__(self):
        return len(self.nameLs)


if __name__ == '__main__':
    root = 'D:/data/lfw_align_112'
    file_list = 'D:/data/pairs.txt'

    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])

    dataset = LFW(root, file_list, transform=transform)
    #dataset = LFW(root, file_list)
    trainloader = data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2, drop_last=False)
    print(len(dataset))
    for data in trainloader:
        for d in data:
            print(d[0].shape)