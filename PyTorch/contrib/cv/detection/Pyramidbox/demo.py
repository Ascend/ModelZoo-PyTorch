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


import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import torch.npu
import cv2
import time
import numpy as np
from PIL import Image

from data.config import cfg
from pyramidbox import build_net
from utils.augmentations import to_chw_bgr

parser = argparse.ArgumentParser(description='pyramidbox demo')
parser.add_argument('--save_dir',
                    type=str, default='tmp/',
                    help='Directory for detect result')
parser.add_argument('--model',
                    type=str, default='/home/wch/Pyramidbox.pytorch-master/weights/pyramidbox_120000_99.02.pth',
                    help='trained model')
parser.add_argument('--thresh',
                    default=0.4, type=float,
                    help='Final confidence threshold')
args = parser.parse_args()


if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

use_npu = torch.npu.is_available()

if use_npu:
    device=f'npu:0'
    torch.npu.set_device(device)
    

def detect(net, img_path, thresh):
    img = Image.open(img_path)
    if img.mode == 'L':
        img = img.convert('RGB')

    img = np.array(img)
    height, width, _ = img.shape
    max_im_shrink = np.sqrt(
        1200 * 1100 / (img.shape[0] * img.shape[1]))
    image = cv2.resize(img, None, None, fx=max_im_shrink,
                       fy=max_im_shrink, interpolation=cv2.INTER_LINEAR)

    x = to_chw_bgr(image)
    x = x.astype('float32')
    x -= cfg.img_mean
    x = x[[2, 1, 0], :, :]
    #x = x * cfg.scale

    x = Variable(torch.from_numpy(x).unsqueeze(0))
    if use_npu:
        x = x.npu()
    t1 = time.time()
    y = net(x)
    detections = y.data
    scale = torch.Tensor([img.shape[1], img.shape[0],
                          img.shape[1], img.shape[0]])

    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= thresh:
            score = detections[0, i, j, 0]
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy().astype(int)
            left_up, right_bottom = (pt[0], pt[1]), (pt[2], pt[3])
            j += 1
            cv2.rectangle(img, left_up, right_bottom, (0, 0, 255), 2)
            conf = "{:.2f}".format(score)
            text_size, baseline = cv2.getTextSize(
                conf, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
            p1 = (left_up[0], left_up[1] - text_size[1])
            cv2.rectangle(img, (p1[0] - 2 // 2, p1[1] - 2 - baseline),
                          (p1[0] + text_size[0], p1[1] + text_size[1]), [255, 0, 0], -1)
            cv2.putText(img, conf, (p1[0], p1[
                        1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, 8)

    t2 = time.time()
    print('detect:{} timer:{}'.format(img_path, t2 - t1))

    cv2.imwrite(os.path.join(args.save_dir, os.path.basename(img_path)), img)


if __name__ == '__main__':
    net = build_net('test', cfg.NUM_CLASSES)
    net.load_state_dict(torch.load(args.model,map_location=lambda storage, loc: storage))
    net.eval()

    if use_npu:
        net.npu()
        cudnn.benckmark = True

    img_path = './img'
    img_list = [os.path.join(img_path, x)
                for x in os.listdir(img_path) if x.endswith('jpg')]
    for path in img_list:
        detect(net, path, args.thresh)
