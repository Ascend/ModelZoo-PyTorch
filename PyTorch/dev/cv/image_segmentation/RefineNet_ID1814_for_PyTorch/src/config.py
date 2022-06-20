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
import numpy as np
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))

# DATASET PARAMETERS
TRAIN_DIR = "./datasets/helen_dataset/"
VAL_DIR = TRAIN_DIR
TRAIN_LIST = ["./data/train.face"] * 3
VAL_LIST = ["./data/val.face"] * 3
SHORTER_SIDE = [352] * 3
CROP_SIZE = [512] * 3
NORMALISE_PARAMS = [
    1.0 / 255,  # SCALE
    np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3)),  # MEAN
    np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3)),
]  # STD
BATCH_SIZE = [6] * 3
NUM_WORKERS = 12
NUM_CLASSES = [11] * 3
LOW_SCALE = [0.5] * 3
HIGH_SCALE = [2.0] * 3
IGNORE_LABEL = 255

# ENCODER PARAMETERS
ENC = "18"
#ENC_PRETRAINED = True  # pre-trained on ImageNet or randomly initialised
ENC_PRETRAINED = False

# GENERAL
EVALUATE = False
FREEZE_BN = [True] * 3
NUM_SEGM_EPOCHS = [100] * 3
PRINT_EVERY = 10
RANDOM_SEED = 42
SNAPSHOT_DIR = "./result_fac/"
CKPT_PATH = "./result_fac/checkpoint.pth.tar"
VAL_EVERY = [100] * 3  # how often to record validation scores

# OPTIMISERS' PARAMETERS
LR_ENC = [5e-4, 2.5e-4, 1e-4]  # TO FREEZE, PUT 0
LR_DEC = [5e-3, 2.5e-3, 1e-3]
# LR_ENC = [5e-4, 2.5e-4, 1e-4]  # TO FREEZE, PUT 0
# LR_DEC = [5e-3, 2.5e-3, 1e-3]
MOM_ENC = [0.9] * 3  # TO FREEZE, PUT 0
MOM_DEC = [0.9] * 3
WD_ENC = [1e-5] * 3  # TO FREEZE, PUT 0
WD_DEC = [1e-5] * 3
OPTIM_DEC = "sgd"
