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
"""
@author: Wenbo Li
@contact: fenglinglwb@gmail.com
"""

import os, getpass
import os.path as osp
import argparse

from easydict import EasyDict as edict
from dataset.attribute import load_dataset
from cvpack.utils.pyt_utils import ensure_dir


class Config:
    # -------- Directoy Config -------- #
    USER = getpass.getuser()
    ROOT_DIR = os.environ['MSPN_HOME']
    OUTPUT_DIR = osp.join(ROOT_DIR, 'model_logs', USER,
            osp.split(osp.split(osp.realpath(__file__))[0])[1])
    TEST_DIR = osp.join(OUTPUT_DIR, 'test_dir')
    TENSORBOARD_DIR = osp.join(OUTPUT_DIR, 'tb_dir') 

    # -------- Data Config -------- #
    DATALOADER = edict()
    DATALOADER.NUM_WORKERS = 4
    DATALOADER.ASPECT_RATIO_GROUPING = False
    DATALOADER.SIZE_DIVISIBILITY = 0

    DATASET = edict()
    DATASET.NAME = 'COCO'
    dataset = load_dataset(DATASET.NAME)
    DATASET.KEYPOINT = dataset.KEYPOINT

    INPUT = edict()
    INPUT.NORMALIZE = True
    INPUT.MEANS = [0.406, 0.456, 0.485] # bgr
    INPUT.STDS = [0.225, 0.224, 0.229]

    # edict will automatcally convert tuple to list, so ..
    INPUT_SHAPE = dataset.INPUT_SHAPE
    OUTPUT_SHAPE = dataset.OUTPUT_SHAPE

    # -------- Model Config -------- #
    MODEL = edict()

    MODEL.BACKBONE = 'Res-50'
    MODEL.UPSAMPLE_CHANNEL_NUM = 256
    MODEL.STAGE_NUM = 2
    MODEL.OUTPUT_NUM = DATASET.KEYPOINT.NUM

    MODEL.DEVICE = 'npu'

    MODEL.WEIGHT = osp.join(ROOT_DIR, 'lib/models/resnet-50_rename.pth')

    # -------- Training Config -------- #
    SOLVER = edict()
    SOLVER.BASE_LR = 5e-4 
    SOLVER.CHECKPOINT_PERIOD = 2400 
    SOLVER.GAMMA = 0.5
    SOLVER.IMS_PER_GPU = 32
    SOLVER.MAX_ITER = 96000 
    SOLVER.MOMENTUM = 0.9
    SOLVER.OPTIMIZER = 'Adam'
    SOLVER.WARMUP_FACTOR = 0.1
    SOLVER.WARMUP_ITERS = 2400 
    SOLVER.WARMUP_METHOD = 'linear'
    SOLVER.WEIGHT_DECAY = 1e-5 
    SOLVER.WEIGHT_DECAY_BIAS = 0

    LOSS = edict()
    LOSS.OHKM = True
    LOSS.TOPK = 8
    LOSS.COARSE_TO_FINE = True

    RUN_EFFICIENT = False 
    # -------- Test Config -------- #
    TEST = dataset.TEST
    TEST.IMS_PER_GPU = 32 


config = Config()
cfg = config


def link_log_dir():
    if not osp.exists('./log'):
        ensure_dir(config.OUTPUT_DIR)
        cmd = 'ln -s ' + config.OUTPUT_DIR + ' log'
        os.system(cmd)


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '-log', '--linklog', default=False, action='store_true')
    return parser


if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    if args.linklog:
        link_log_dir()
