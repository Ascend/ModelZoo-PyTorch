# -*- coding: utf-8 -*-
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

import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from skimage.io import imread, imsave

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms

from dataset import Dataset

import archs
from metrics import dice_coef, batch_iou, mean_iou, iou_score
import losses
from utils import str2bool, count_params


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name')

    args = parser.parse_args()

    return args


def main():
    val_args = parse_args()

    args = joblib.load('models/%s/args.pkl' %val_args.name)

    if not os.path.exists('output/%s' %args.name):
        os.makedirs('output/%s' %args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    joblib.dump(args, 'models/%s/args.pkl' %args.name)

    # create model
    print("=> creating model %s" %args.arch)
    model = archs.__dict__[args.arch](args)

    model = model.npu()

    # Data loading code
    img_paths = glob('input/' + args.dataset + '/images/*')
    mask_paths = glob('input/' + args.dataset + '/masks/*')

    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = \
        train_test_split(img_paths, mask_paths, test_size=0.2, random_state=41)

    model.load_state_dict(torch.load('models/%s/model.pth' %args.name))
    model.eval()

    val_dataset = Dataset(args, val_img_paths, val_mask_paths)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        with torch.no_grad():
            for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
                input = input.npu()
                target = target.npu()

                # compute output
                if args.deepsupervision:
                    output = model(input)[-1]
                else:
                    output = model(input)

                output = torch.sigmoid(output).data.cpu().numpy()
                img_paths = val_img_paths[args.batch_size*i:args.batch_size*(i+1)]

                for i in range(output.shape[0]):
                    imsave('output/%s/'%args.name+os.path.basename(img_paths[i]), (output[i,0,:,:]*255).astype('uint8'))

        torch.npu.empty_cache()

    # IoU
    ious = []
    for i in tqdm(range(len(val_mask_paths))):
        mask = imread(val_mask_paths[i])
        pb = imread('output/%s/'%args.name+os.path.basename(val_mask_paths[i]))

        mask = mask.astype('float32') / 255
        pb = pb.astype('float32') / 255

        '''
        plt.figure()
        plt.subplot(121)
        plt.imshow(mask)
        plt.subplot(122)
        plt.imshow(pb)
        plt.show()
        '''

        iou = iou_score(pb, mask)
        ious.append(iou)
    print('IoU: %.4f' %np.mean(ious))


if __name__ == '__main__':
    main()
