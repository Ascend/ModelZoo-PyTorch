#-*- coding:utf-8 -*-
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
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import argparse
import os
from data.config import cfg
#import cv2

parser = argparse.ArgumentParser(
    description='Pyramidbox face Detector Training With Pytorch')
parser.add_argument('--data_path',
                    default=None, type=str,
                    help='data_path')
args = parser.parse_args()

train_list_file = os.path.join(args.data_path, 'wider_face_split',
                               'wider_face_train_bbx_gt.txt')
val_list_file = os.path.join(args.data_path, 'wider_face_split',
                             'wider_face_val_bbx_gt.txt')

WIDER_TRAIN = os.path.join(args.data_path, 'WIDER_train', 'images')
WIDER_VAL = os.path.join(args.data_path, 'WIDER_val', 'images')


def parse_wider_file(root, file):
    with open(file, 'r') as fr:
        lines = fr.readlines()
    face_count = []
    img_paths = []
    face_loc = []
    img_faces = []
    count = 0
    flag = False
    for k, line in enumerate(lines):
        line = line.strip().strip('\n')
        if count > 0:
            line = line.split(' ')
            count -= 1
            loc = [int(line[0]), int(line[1]), int(line[2]), int(line[3])]
            face_loc += [loc]
        if flag:
            face_count += [int(line)]
            flag = False
            count = int(line)
        if 'jpg' in line:
            img_paths += [os.path.join(root, line)]
            flag = True

    total_face = 0
    for k in face_count:
        face_ = []
        for x in range(total_face, total_face + k):
            face_.append(face_loc[x])
        img_faces += [face_]
        total_face += k
    return img_paths, img_faces


def wider_data_file():
    img_paths, bbox = parse_wider_file(WIDER_TRAIN, train_list_file)
    fw = open(cfg.FACE.TRAIN_FILE, 'w')
    for index in range(len(img_paths)):
        path = img_paths[index]
        boxes = bbox[index]
        fw.write(path)
        fw.write(' {}'.format(len(boxes)))
        for box in boxes:
            data = ' {} {} {} {} {}'.format(box[0], box[1], box[2], box[3], 1)
            fw.write(data)
        fw.write('\n')
    fw.close()

    img_paths, bbox = parse_wider_file(WIDER_VAL, val_list_file)
    fw = open(cfg.FACE.VAL_FILE, 'w')
    for index in range(len(img_paths)):
        path = img_paths[index]
        boxes = bbox[index]
        fw.write(path)
        fw.write(' {}'.format(len(boxes)))
        for box in boxes:
            data = ' {} {} {} {} {}'.format(box[0], box[1], box[2], box[3], 1)
            fw.write(data)
        fw.write('\n')
    fw.close()


if __name__ == '__main__':
    wider_data_file()
