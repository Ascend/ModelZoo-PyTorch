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
"""
@author: Wenbo Li
@contact: fenglinglwb@gmail.com
"""

from easydict import EasyDict as edict

class COCO:
    NAME = 'COCO'

    KEYPOINT = edict()
    KEYPOINT.NUM = 17
    KEYPOINT.FLIP_PAIRS = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12],
            [13, 14], [15, 16]]
    KEYPOINT.UPPER_BODY_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    KEYPOINT.LOWER_BODY_IDS = [11, 12, 13, 14, 15, 16]
    KEYPOINT.LOAD_MIN_NUM = 1

    INPUT_SHAPE = (256, 192) # height, width
    OUTPUT_SHAPE = (64, 48)
    WIDTH_HEIGHT_RATIO = INPUT_SHAPE[1] / INPUT_SHAPE[0]

    PIXEL_STD = 200
    COLOR_RGB = False

    TRAIN = edict()
    TRAIN.BASIC_EXTENTION = 0.05
    TRAIN.RANDOM_EXTENTION = True
    TRAIN.X_EXTENTION = 0.6 
    TRAIN.Y_EXTENTION = 0.8 
    TRAIN.SCALE_FACTOR_LOW = -0.25 
    TRAIN.SCALE_FACTOR_HIGH = 0.25 
    TRAIN.SCALE_SHRINK_RATIO = 0.8
    TRAIN.ROTATION_FACTOR = 45 
    TRAIN.PROB_ROTATION = 0.5
    TRAIN.PROB_FLIP = 0.5
    TRAIN.NUM_KEYPOINTS_HALF_BODY = 3
    TRAIN.PROB_HALF_BODY = 0.3
    TRAIN.X_EXTENTION_HALF_BODY = 0.6 
    TRAIN.Y_EXTENTION_HALF_BODY = 0.8
    TRAIN.ADD_MORE_AUG = False
    TRAIN.GAUSSIAN_KERNELS = [(15, 15), (11, 11), (9, 9), (7, 7), (5, 5)]

    TEST = edict()
    TEST.FLIP = True
    TEST.X_EXTENTION = 0.01 * 9.0
    TEST.Y_EXTENTION = 0.015 * 9.0
    TEST.SHIFT_RATIOS = [0.25]
    TEST.GAUSSIAN_KERNEL = 5


class MPII:
    NAME = 'MPII'

    KEYPOINT = edict()
    KEYPOINT.NUM = 16
    KEYPOINT.FLIP_PAIRS = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]
    KEYPOINT.UPPER_BODY_IDS = [7, 8, 9, 10, 11, 12, 13, 14, 15]
    KEYPOINT.LOWER_BODY_IDS = [0, 1, 2, 3, 4, 5, 6]
    KEYPOINT.LOAD_MIN_NUM = 1

    INPUT_SHAPE = (256, 256) # height, width
    OUTPUT_SHAPE = (64, 64)
    WIDTH_HEIGHT_RATIO = INPUT_SHAPE[1] / INPUT_SHAPE[0]

    PIXEL_STD = 200
    COLOR_RGB = False

    TRAIN = edict()
    TRAIN.BASIC_EXTENTION = 0.0
    TRAIN.RANDOM_EXTENTION = False 
    TRAIN.X_EXTENTION = 0.25 
    TRAIN.Y_EXTENTION = 0.25
    TRAIN.SCALE_FACTOR_LOW = -0.25 
    TRAIN.SCALE_FACTOR_HIGH = 0.25 
    TRAIN.SCALE_SHRINK_RATIO = 1.0
    TRAIN.ROTATION_FACTOR = 60
    TRAIN.PROB_ROTATION = 0.5
    TRAIN.PROB_FLIP = 0.5
    TRAIN.NUM_KEYPOINTS_HALF_BODY = 8
    TRAIN.PROB_HALF_BODY = 0.5
    TRAIN.X_EXTENTION_HALF_BODY = 0.6 
    TRAIN.Y_EXTENTION_HALF_BODY = 0.6
    TRAIN.ADD_MORE_AUG = False
    TRAIN.GAUSSIAN_KERNELS = [(15, 15), (11, 11), (9, 9), (7, 7), (5, 5)]

    TEST = edict()
    TEST.FLIP = True
    TEST.X_EXTENTION = 0.25 
    TEST.Y_EXTENTION = 0.25 
    TEST.SHIFT_RATIOS = [0.25]
    TEST.GAUSSIAN_KERNEL = 9


def load_dataset(name):
    if name == 'COCO':
        dataset = COCO()
    elif name == 'MPII':
        dataset = MPII()
    return dataset
