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
from easydict import EasyDict
import numpy as np


class Config(object):
    # data augument config
    expand_prob = 0.5
    expand_max_ratio = 4
    hue_prob = 0.5
    hue_delta = 18
    contrast_prob = 0.5
    contrast_delta = 0.5
    saturation_prob = 0.5
    saturation_delta = 0.5
    brightness_prob = 0.5
    brightness_delta = 0.125
    data_anchor_sampling_prob = 0.5
    min_face_size = 6.0
    apply_distort = True
    apply_expand = False
    img_mean = np.array([104., 117., 123.])[:, np.newaxis, np.newaxis].astype(
        'float32')
    resize_width = 640
    resize_height = 640
    scale = 1 / 127.0
    anchor_sampling = True
    filter_min_face = True

    # train config
    LR_STEPS = (80000,100000,120000)
    MAX_STEPS = 150000
    EPOCHES = 100

    # anchor config
    FEATURE_MAPS = [160, 80, 40, 20, 10, 5]
    INPUT_SIZE = 640
    STEPS = [4, 8, 16, 32, 64, 128]
    ANCHOR_SIZES1 = [8, 16, 32, 64, 128, 256]
    ANCHOR_SIZES2 = [16, 32, 64, 128, 256, 512]
    ASPECT_RATIO = [1.0]
    CLIP = False
    VARIANCE = [0.1, 0.2]

    # detection config
    NMS_THRESH = 0.3
    NMS_TOP_K = 5000
    TOP_K = 750
    CONF_THRESH = 0.05

    # loss config
    NEG_POS_RATIOS = 3
    NUM_CLASSES = 2

    #multigpu
    MultiGPU_ID =[0,1,2,3,4,5,6,7]

    # dataset config
    HOME = '/opt/npu/dataset/WIDERFace/'

    # face config
    FACE = EasyDict()
    FACE_TRAIN_FILE = './val_data/face_train.txt' #进行训练图片集合，由 prepare_wide_data.pyd得到
    FACE_VAL_FILE = './val_data/face_val.txt' #进行验证图片集合
    FACE_FDDB_DIR = ''
    FACE_WIDER_DIR = '/opt/npu/dataset/WIDERFace'
    FACE_AFW_DIR = ''
    FACE_PASCAL_DIR = ''
    FACE.OVERLAP_THRESH = 0.35

cur_config = Config()