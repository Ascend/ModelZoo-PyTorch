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
## DEFAULT CONFIGURATION USED IN OUR EXPERIMENTS ON 2 GPUs

import numpy as np
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))

# DATASET CONFIGURATION
BATCH_SIZE = [64, 32]
META_TRAIN_PRCT = 90
NORMALISE_PARAMS = [
    1.0 / 255,  # SCALE
    np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3)),  # MEAN
    np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3)),  # STD
]
NUM_CLASSES = [21, 21]
NUM_WORKERS = 16
N_TASK0 = 4000  # store in-memory these many samples for the first task
TRAIN_DIR = "./data/datasets/VOCdevkit/"
TRAIN_LIST = "./data/lists/train+.lst"
VAL_BATCH_SIZE = 64
VAL_CROP_SIZE = 400
VAL_DIR = "./data/datasets/VOCdevkit/"
VAL_LIST = "./data/lists/train+.lst"  # meta-train and meta-val learning
VAL_OMIT_CLASSES = [0]  # ignore background when computing the reward
VAL_RESIZE_SIDE = 400

# AUGMENTATIONS CONFIGURATION
CROP_SIZE = [256, 350]
HIGH_SCALE = 1.4
LOW_SCALE = 0.7
RESIZE_SIDE = [300, 400]

# ENCODER OPTIMISATION CONFIGURATION
ENC_GRAD_CLIP = 3.0
ENC_LR = [1e-3, 1e-3]
ENC_MOM = [0.9] * 3
ENC_OPTIM = "sgd"
ENC_WD = [1e-5] * 3

# DECODER OPTIMISATION CONFIGURATION
DEC_AUX_WEIGHT = 0.15  # to disable aux, set to -1
DEC_GRAD_CLIP = 3.0
DEC_LR = [3e-3, 3e-3]
DEC_MOM = [0.9] * 3
DEC_OPTIM = "adam"
DEC_WD = [1e-5] * 3

# GENERAL OPTIMISATION CONFIGURATION
DO_KD = True
DO_POLYAK = True
FREEZE_BN = [False, False]
KD_COEFF = 0.3
NUM_EPOCHS = 20000
NUM_SEGM_EPOCHS = [5, 1]
RANDOM_SEED = 9314
VAL_EVERY = [5, 1]  # how often to record validation scores

# GENERAL DEBUGGING CONFIGURATION
CKPT_PATH = "./ckpt/checkpoint.pth.tar"
PRINT_EVERY = 20
SNAPSHOT_DIR = "./ckpt/"
SUMMARY_DIR = "./tb_logs/"

# CONTROLLER CONFIGURATION: USED TO CREATE RNN-BASED CONTROLLER
CELL_MAX_REPEAT = 4
CELL_MAX_STRIDE = 2
CELL_NUM_LAYERS = 4
CTRL_AGENT = "ppo"
CTRL_BASELINE_DECAY = 0.95
CTRL_LR = 1e-4
CTRL_VERSION = "cvpr"
DEC_NUM_CELLS = 3
LSTM_HIDDEN_SIZE = 100
LSTM_NUM_LAYERS = 2
NUM_AGG_OPS = 2
NUM_OPS = 11

# DECODER CONFIGURATION: USED TO CREATE DECODER ARCHITECTURE
AGG_CELL_SIZE = 48
AUX_CELL = True
SEP_REPEATS = 1
