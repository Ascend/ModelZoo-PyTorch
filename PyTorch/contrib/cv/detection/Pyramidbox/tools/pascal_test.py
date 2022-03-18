#-*- coding:utf-8 -*-
# Copyright 2021 Huawei Technologies Co., Ltd
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
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import torch
import argparse
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import os.path as osp

import cv2
import time
import numpy as np
from PIL import Image

from data.config import cfg
from pyramidbox import build_net
from torch.autograd import Variable
from utils.augmentations import to_chw_bgr

parser = argparse.ArgumentParser(description='pyramidbox evaluatuon pascal')
parser.add_argument('--model', 
                    type=str,default='weights/pyramidbox.pth', 
                    help='trained model')
parser.add_argument('--thresh', 
                    default=0.1, type=float,
                    help='Final confidence threshold')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()

if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

PASCAL_IMG_DIR = os.path.join(cfg.FACE.PASCAL_DIR, 'images')
PASCAL_RESULT_DIR = os.path.join(cfg.FACE.PASCAL_DIR, 'pyramidbox')
PASCAL_RESULT_IMG_DIR = os.path.join(PASCAL_RESULT_DIR, 'images')

if not os.path.exists(PASCAL_RESULT_IMG_DIR):
    os.makedirs(PASCAL_RESULT_IMG_DIR)


def detect_face(net, img, thresh):
    height, width, _ = img.shape
    im_shrink = 640.0 / max(height, width)
    image = cv2.resize(img, None, None, fx=im_shrink,
                       fy=im_shrink, interpolation=cv2.INTER_LINEAR).copy()

    x = to_chw_bgr(image)
    x = x.astype('float32')
    x -= cfg.img_mean
    x = x[[2, 1, 0], :, :]

    x = Variable(torch.from_numpy(x).unsqueeze(0))
    if use_cuda:
        x = x.cuda()

    y = net(x)
    detections = y.data
    scale = torch.Tensor([img.shape[1], img.shape[0],
                          img.shape[1], img.shape[0]])

    bboxes = []
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= thresh:
            box = []
            score = detections[0, i, j, 0]
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy().astype(np.int)
            j += 1
            box += [pt[0], pt[1], pt[2], pt[3], score]
            box[1] += 0.2 * (box[3] - box[1] + 1)
            bboxes += [box]

    return bboxes


if __name__ == '__main__':
    net = build_net('test', cfg.NUM_CLASSES)
    net.load_state_dict(torch.load(args.model))
    net.eval()

    if use_cuda:
        net.cuda()
        cudnn.benckmark = True

    #transform = S3FDBasicTransform(cfg.INPUT_SIZE, cfg.MEANS)

    counter = 0
    txt_out = os.path.join(PASCAL_RESULT_DIR, 'pyramidbox_dets.txt')
    txt_in = os.path.join('./tools/pascal_img_list.txt')

    fout = open(txt_out, 'w')
    fin = open(txt_in, 'r')

    for line in fin.readlines():
        line = line.strip()
        img_file = os.path.join(PASCAL_IMG_DIR, line)
        out_file = os.path.join(PASCAL_RESULT_IMG_DIR, line)
        counter += 1
        t1 = time.time()
        #img = cv2.imread(img_file, cv2.IMREAD_COLOR)
        img = Image.open(img_file)
        if img.mode == 'L':
            img = img.convert('RGB')
        img = np.array(img)
        bboxes = detect_face(net, img, args.thresh)
        t2 = time.time()
        print('Detect %04d th image costs %.4f' % (counter, t2 - t1))
        for bbox in bboxes:
            x1, y1, x2, y2, score = bbox
            fout.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(
                line, score, x1, y1, x2, y2))
        for bbox in bboxes:
            x1, y1, x2, y2, score = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(out_file, img)

    fout.close()
    fin.close()
