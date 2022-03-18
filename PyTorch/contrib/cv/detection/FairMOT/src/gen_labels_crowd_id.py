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

import os.path as osp
import os
import cv2
import json
import numpy as np


def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)


def load_func(fpath):
    print('fpath', fpath)
    assert os.path.exists(fpath)
    with open(fpath, 'r') as fid:
        lines = fid.readlines()
    records =[json.loads(line.strip('\n')) for line in lines]
    return records


def gen_labels_crowd(data_root, label_root, ann_root):
    mkdirs(label_root)
    anns_data = load_func(ann_root)

    tid_curr = 0
    for i, ann_data in enumerate(anns_data):
        print(i)
        image_name = '{}.jpg'.format(ann_data['ID'])
        img_path = os.path.join(data_root, image_name)
        anns = ann_data['gtboxes']
        img = cv2.imread(
            img_path,
            cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
        img_height, img_width = img.shape[0:2]
        for i in range(len(anns)):
            if 'extra' in anns[i] and 'ignore' in anns[i]['extra'] and anns[i]['extra']['ignore'] == 1:
                continue
            x, y, w, h = anns[i]['fbox']
            x += w / 2
            y += h / 2
            label_fpath = img_path.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt')
            label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                tid_curr, x / img_width, y / img_height, w / img_width, h / img_height)
            with open(label_fpath, 'a') as f:
                f.write(label_str)
            tid_curr += 1


if __name__ == '__main__':
    data_val = '/data/yfzhang/MOT/JDE/crowdhuman/images/val'
    label_val = '/data/yfzhang/MOT/JDE/crowdhuman/labels_with_ids/val'
    ann_val = '/data/yfzhang/MOT/JDE/crowdhuman/annotation_val.odgt'
    data_train = '/data/yfzhang/MOT/JDE/crowdhuman/images/train'
    label_train = '/data/yfzhang/MOT/JDE/crowdhuman/labels_with_ids/train'
    ann_train = '/data/yfzhang/MOT/JDE/crowdhuman/annotation_train.odgt'
    gen_labels_crowd(data_train, label_train, ann_train)
    gen_labels_crowd(data_val, label_val, ann_val)


