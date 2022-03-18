# Copyright 2021 Huawei Technologies Co., Ltd
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

#coding=utf-8

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

import cv2
import time
import numpy as np
from PIL import Image

from data.config import cur_config as cfg
from models.factory import build_net
from torch.autograd import Variable
from utils.augmentations import to_chw_bgr


parser = argparse.ArgumentParser(description='dsfd demo')
parser.add_argument('--network',
                    default='resnet152', type=str,
                    choices=['vgg', 'resnet50', 'resnet101', 'resnet152'],
                    help='model for training')
parser.add_argument('--save_dir',
                    type=str, default='tmp/',
                    help='Directory for detect result')
parser.add_argument('--model', type=str, default='weights/resnet152/dsfd_20000.pth', help='trained model')
parser.add_argument('--thresh',
                    default=0.6, type=float,
                    help='Final confidence threshold')
args = parser.parse_args()


if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

#use_cuda = torch.cuda.is_available()
#use_npu = torch.npu.is_available()


torch.set_default_tensor_type('torch.FloatTensor')


def detect(net, img_path, thresh):
    img = Image.open(img_path)
    if img.mode == 'L':
        img = img.convert('RGB')

    img = np.array(img)
    height, width, _ = img.shape
    max_im_shrink = np.sqrt(
        1500 * 1000 / (img.shape[0] * img.shape[1]))
    image = cv2.resize(img, None, None, fx=max_im_shrink,
                       fy=max_im_shrink, interpolation=cv2.INTER_LINEAR)

    x = to_chw_bgr(image)
    x = x.astype('float32')
    x -= cfg.img_mean
    x = x[[2, 1, 0], :, :]

    x = Variable(torch.from_numpy(x).unsqueeze(0))
    #if use_npu:
    #    x = x.npu()
    t1 = time.time()
    y = net(x)
    detections = y.data
    scale = torch.Tensor([img.shape[1], img.shape[0],
                          img.shape[1], img.shape[0]])

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    print("get detections.size(1):", detections.size(1))
    for i in range(detections.size(1)):
        j = 0
        print("get ori score:", detections[0, i, j, 0])
        count = 0
        while detections[0, i, j, 0] >= thresh:
            score = detections[0, i, j, 0]
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy().astype(int)
            left_up, right_bottom = (pt[0], pt[1]), (pt[2], pt[3])
            print("get left_up:", left_up)
            print("get right_bottom:", right_bottom)
            j += 1
            face_w = right_bottom[0] - left_up[0]
            face_h = right_bottom[1] - left_up[1]
            print("get face w,h:", face_w, face_h)       
            cv2.rectangle(img, left_up, right_bottom, (0, 0, 255), 2)
            conf = "{:.2f}".format(score)
            print("get conf:", conf)
            text_size, baseline = cv2.getTextSize(
                conf, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
            p1 = (left_up[0], left_up[1] - text_size[1])
            #cv2.rectangle(img, (p1[0] - 2 // 2, p1[1] - 2 - baseline),
            #              (p1[0] + text_size[0], p1[1] + text_size[1]),[255,0,0], -1)
            cv2.putText(img, conf, (p1[0], p1[
                            1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, 8)
            count += 1
            print("get count:", count)
            if count >= 100:
                print("count >= 100")
                break

    t2 = time.time()
    print('detect:{} timer:{}'.format(img_path, t2 - t1))

    cv2.imwrite(os.path.join(args.save_dir, os.path.basename(img_path)), img)


if __name__ == '__main__':
    net = build_net('test', cfg.NUM_CLASSES, args.network)
    net.load_state_dict(torch.load(args.model, map_location='cpu'))
    net.eval()

    #if use_npu:
    #   net.npu()

    img_path = './img'
    img_list = [os.path.join(img_path, x)
                for x in os.listdir(img_path) if x.endswith('jpg')]
    for path in img_list:
        detect(net, path, args.thresh)
