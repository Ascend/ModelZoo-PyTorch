# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#-*- coding:utf-8 -*-
import os

#img_dir = '../imagedata/image/'
#label_dir = '../imagedata/xml/'

img_dir = '/home/dockerHome/ctpn/ctpn_8p/imagedata/Challenge2_Training_Task12_Images/'
label_dir = '/home/dockerHome/ctpn/ctpn_8p/imagedata/Challenge2_Training_Task1_GT/'

num_workers = 0
pretrained_weights = ''
#pretrained_weights = './checkpoints/gpu_ctpn_ep98_0.2615_0.0333_0.2948.pth'


anchor_scale = 16
IOU_NEGATIVE = 0.3
IOU_POSITIVE = 0.7
IOU_SELECT = 0.7

RPN_POSITIVE_NUM = 150
RPN_TOTAL_NUM = 300

IMAGE_MEAN = [123.68, 116.779, 103.939]

# online hard example mining
OHEM = True
checkpoints_dir = './checkpoints'
