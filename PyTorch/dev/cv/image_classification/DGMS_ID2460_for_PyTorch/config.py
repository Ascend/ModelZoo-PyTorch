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
""" Global configurations file.
"""

# Dataset settings
NUM_CLASSES = {
    'cifar10': 10,
    'imagenet': 1000,
    'cub200': 200,
    'cars': 196,
    'aircraft': 100,
}

DATA_FOLDERS = {
    'cifar': 'Path2DatasetCIFAR10/',
    'imagenet': 'Path2DatasetImageNet/',
    'cub200': 'Path2DatasetCUB_200_2011/',
    'cars': 'Path2DatasetStanfordCars/',
    'aircraft': 'Path2DatasetFGVCAircraft/',
}

MEANS = {
    'cifar': (0.4914, 0.4822, 0.4465),
    'imagenet': (0.485, 0.456, 0.406),
    'cub200': (0.485, 0.456, 0.406),
    'cars': (0.485, 0.456, 0.406),
    'aircraft': (0.485, 0.456, 0.406),
}

STDS = {
    'cifar': (0.2023, 0.1994, 0.2010),
    'imagenet': (0.229, 0.224, 0.225),
    'cub200': (0.229, 0.224, 0.225),
    'cars': (0.229, 0.224, 0.225),
    'aircraft': (0.229, 0.224, 0.225),
}

# Model definition
TAU = 0.01
IS_TRAIN = True
K_LEVEL = 16
IS_NORMAL = True
IS_EMP = False

# Training settings
BATCH_SIZE = {
    'cifar10': 128,
    'imagenet': 256,
    'cub200': 256,
    'cars': 256,
    'aircraft': 256,
}

EPOCH = {
    'cifar10': 350,
    'imagenet': 60,
    'cub200': 60,
    'cars': 60,
    'aircraft': 60,
}

LAYER = {
    'resnet20': 20,
    'resnet32': 32,
    'resnet56': 56,
    'vggsmall': 7,
    'resnet18': 21,
    'resnet50': 54,
    'mnasnet': 53,
    'proxylessnas': 62,
}

L_CNT = 0
LAYER_NUM = 20
EPS = 1e-11
KEEP = True
DEBUG = False
SKIPPED_LAYERS = []

def set_status(flag):
    global IS_TRAIN
    IS_TRAIN = flag

def count_layer():
    global L_CNT
    L_CNT = L_CNT + 1


def set_config(args):
    global IS_EMP, IS_NORMAL, K_LEVEL, TAU, LAYER, LAYER_NUM, SKIPPED_LAYERS
    IS_EMP = args.empirical
    IS_NORMAL = args.normal
    TAU = args.tau
    K_LEVEL = args.K
    LAYER_NUM = LAYER[args.network]
    SKIPPED_LAYERS = [1, LAYER_NUM]