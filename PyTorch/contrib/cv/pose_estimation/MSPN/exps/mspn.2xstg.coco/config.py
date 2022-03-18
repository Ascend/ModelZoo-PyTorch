# encoding: utf-8
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
# ============================================================================
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
    #######npu8P modify1 begin############################
    os.environ['MASTER_ADDR'] = '127.0.0.1' # 可以使用当前真实ip或者'127.0.0.1'
    os.environ['MASTER_PORT'] = '29688' # 随意一个可使用的port即可
    #######npu8P modify1 end###############################
    
    # -------- Directoy Config -------- #
    USER = getpass.getuser()
    ROOT_DIR = os.environ['MSPN_HOME']
    OUTPUT_DIR = osp.join(ROOT_DIR, 'model_logs', USER,
            osp.split(osp.split(osp.realpath(__file__))[0])[1])
    TEST_DIR = osp.join(OUTPUT_DIR, 'test_dir')
    TENSORBOARD_DIR = osp.join(OUTPUT_DIR, 'tb_dir') 

    # -------- Data Config -------- #
    DATALOADER = edict()
    DATALOADER.NUM_WORKERS = 24
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
    SOLVER.OPTIMIZER = 'NpuFusedAdam'
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
