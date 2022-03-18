#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
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

# config.py
import os.path

# gets home dir cross platform
# HOME = os.path.expanduser("~")
HOME = '/home/ljh/refinedet/data'

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)

# SSD CONFIGS
voc = {
    '300': {
        'amp':True,
        'num_classes': 21,
        'lr_steps': (80000, 100000, 120000),
        'max_iter': 121000,
        'feature_maps': [38, 19, 10, 5, 3, 1],
        'min_dim': 300,
        'steps': [8, 16, 32, 64, 100, 300],
        'min_sizes': [30, 60, 111, 162, 213, 264],
        'max_sizes': [60, 111, 162, 213, 264, 315],
        'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        'variance': [0.1, 0.2],
        'clip': True,
        'name': 'VOC_300',
    },
    '512': {
        'amp':True,
        'num_classes': 21,
        'lr_steps': (80000, 100000, 120000),
        'max_iter': 121000,
        'feature_maps': [64, 32, 16, 8, 4, 2, 1],
        'min_dim': 512,
        'steps': [8, 16, 32, 64, 128, 256, 512],
        'min_sizes': [20, 51, 133, 215, 296, 378, 460],
        'max_sizes': [51, 133, 215, 296, 378, 460, 542],
        'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
        'variance': [0.1, 0.2],
        'clip': True,
        'name': 'VOC_512',
    }
}

coco = {
    'amp':True,
    'num_classes': 201,
    'lr_steps': (280000, 360000, 400000),
    'max_iter': 400000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [21, 45, 99, 153, 207, 261],
    'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'COCO',
}

# RefineDet CONFIGS
voc_refinedet = {
    '320': {
        'amp':True,
        'num_classes': 21,
        'rank':0,
        'distributed': True,
        'num_epochs': 232,
        'lr_steps': (80000, 100000, 120000),
        'lr_step_epoch':(154, 193, 232),
        'confidence_threshold': 0.01,
        'top_k': 5,
        'save_folder': 'eval/',
        'cuda': False,
        'npu': True,
        'cleanup': True,
        'input_size': 320,
        'max_iter': 125000,
        'feature_maps': [40, 20, 10, 5],
        'min_dim': 320,
        'steps': [8, 16, 32, 64],
        'min_sizes': [32, 64, 128, 256],
        'max_sizes': [],
        'aspect_ratios': [[2], [2], [2], [2]],
        'variance': [0.1, 0.2],
        'clip': True,
        'name': 'RefineDet_VOC_320',
    },
    '512': {
        'amp':True,
        'num_classes': 21,
        'lr_steps': (80000, 100000, 120000),
        'epoch_steps': (153, 193, 232),
        'max_iter': 121000,
        'feature_maps': [64, 32, 16, 8],
        'min_dim': 512,
        'steps': [8, 16, 32, 64],
        'min_sizes': [32, 64, 128, 256],
        'max_sizes': [],
        'aspect_ratios': [[2], [2], [2], [2]],
        'variance': [0.1, 0.2],
        'clip': True,
        'name': 'RefineDet_VOC_512',
    }
}

coco_refinedet = {
    'amp':True,
    'num_classes': 201,
    'lr_steps': (280000, 360000, 400000),
    'epoch_steps': (153, 193, 232),
    'max_iter': 400000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [21, 45, 99, 153, 207, 261],
    'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'COCO',
}

