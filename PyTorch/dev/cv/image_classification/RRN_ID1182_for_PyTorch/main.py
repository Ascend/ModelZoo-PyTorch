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
from __future__ import print_function
import argparse
from math import log10
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import get_training_set 
import pdb
from torch.optim import lr_scheduler 
import socket
import time
import cv2
import math
import sys
from utils import Logger 
import numpy as np
from arch import RRN
import datetime
import torchvision.utils as vutils
import random
from loss import CharbonnierLoss
import torch.npu
import os
import apex
from apex import amp
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')

parser = argparse.ArgumentParser(description='PyTorch RRN')
parser.add_argument('--scale', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--batchsize', type=int, default=4, help='training batch size')
parser.add_argument('--start_epoch', type=int, default=1, help='Starting epoch for continuing training')
parser.add_argument('--nEpochs', type=int, default=70, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=10, help='Snapshots. This is a savepoint, using to save training model.')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.01')
parser.add_argument('--threads', type=int, default=32, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=0, help='random seed to use. Default=0')
parser.add_argument('--file_list', type=str, default='sep_trainlist.txt', help='where record all of image name in dataset.')
parser.add_argument('--patch_size', type=int, default=64, help='0 to use original frame size')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--layer', type=int, default=10, help='network layer')
parser.add_argument('--stepsize', type=int, default=60, help='Learning rate is decayed by a factor of 10 every half of total epochs')
parser.add_argument('--gamma', type=float, default=0.1 , help='learning rate decay')
parser.add_argument('--save_model_path', type=str, default='./result/weight', help='Location to save checkpoint models')
parser.add_argument('--save_train_log', type=str ,default='./result/log/')
parser.add_argument('--weight-decay', default=5e-04, type=float,help="weight decay (default: 5e-04)")
parser.add_argument('--log_name', type=str, default='rrn-10')
parser.add_argument('--other_dataset', type=bool, default=False, help="If True using vid4k(test),else using vimo90k")
parser.add_argument('--gpu-devices', default='0,1,2,3', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES') 
parser.add_argument('--apex', action='store_true',help='Use apex for mixed precision training')
parser.add_argument('--apex-opt-level', default='O2', type=str,
                    help='For apex mixed precision training'
                         'O0 for FP32 training, O1 for mixed precision training.')
parser.add_argument('--loss-scale-value', default=128., type=float,
                    help='loss scale using in amp, default -1 means dynamic')
opt = parser.parse_args()
opt.data_dir = '/home/panj/data/vimeo_crop_256/sequences'
systime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
def main():
    torch.manual_seed(opt.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_devices
    sys.stdout = Logger(os.path.join(opt.save_train_log, 'train_'+opt.log_name+'.txt'))
    if not torch.npu.is_available():
        raise Exception('No Gpu found, please run with gpu')
    else:
       use_gpu = torch.npu.is_available()
    if use_gpu:
        cudnn.benchmark = True
        torch.npu.manual_seed_all(opt.seed)
    pin_memory = True if use_gpu else False

    print(opt)
    print('===> Loading Datasets')
    train_set = get_training_set(opt.data_dir, opt.scale, opt.data_augmentation, opt.file_list) 
    train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchsize, shuffle=True, pin_memory=pin_memory, drop_last=True)
    print('===> DataLoading Finished')
    # Selecting network layer
    n_c = 128
    n_b = 10
    rrn = RRN(opt.scale, n_c, n_b) # initial filter generate network 
    p = sum(p.numel() for p in rrn.parameters())*4/1048576.0
    print('Model Size: {:.2f}M'.format(p))
    print(rrn)
    print('===> {}L model has been initialized'.format(n_b))
    rrn = rrn
    criterion = nn.L1Loss(reduction='sum')
    if use_gpu:
        rrn = rrn.npu()
        criterion = criterion.npu()
    optimizer = apex.optimizers.NpuFusedAdam(rrn.parameters(), lr = opt.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=opt.weight_decay) 
    if opt.apex:
        rrn, optimizer = amp.initialize(rrn, optimizer,
                                        opt_level=opt.apex_opt_level,
                                        loss_scale=opt.loss_scale_value,
                                        combine_grad=True)
    if opt.stepsize > 0:
        scheduler = lr_scheduler.StepLR(optimizer, step_size = opt.stepsize, gamma=opt.gamma)

    for epoch in range(opt.start_epoch, opt.nEpochs+1):
        train(train_loader, rrn, opt.scale, criterion, optimizer, epoch, use_gpu, n_c) #fed data into network
        scheduler.step()
        if (epoch) % (opt.snapshots) == 0:
            checkpoint(rrn, epoch)

def train(train_loader, rrn, scale, criterion, optimizer, epoch, use_gpu, n_c):
    train_mode = True
    epoch_loss = 0
    rrn.train()
    n=0
    for iteration, data in enumerate(train_loader):
        if n==5:
            pass
        n=n+1
        x_input, target = data[0], data[1] # input and target are both tensor, input:[N,C,T,H,W] , target:[N,C,H,W]
        if use_gpu:
            x_input = Variable(x_input).npu()
            target = Variable(target).npu()
        t0 = time.time()
        optimizer.zero_grad()
        B, _, T, _ ,_ = x_input.shape
        out = []
        init = True
        for i in range(T-1):
            if init:
                init_temp = torch.zeros_like(x_input[:,0:1,0,:,:])
                init_o = init_temp.repeat(1, scale*scale*3,1,1)
                init_h = init_temp.repeat(1, n_c, 1,1)
                h, prediction = rrn(x_input[:,:,i:i+2,:,:], init_h, init_o, init)
                out.append(prediction)
                init = False
            else:
                h, prediction = rrn(x_input[:,:,i:i+2,:,:], h, prediction, init)
                out.append(prediction)
        prediction = torch.stack(out, dim=2)
        loss = criterion(prediction, target)/(B*T)
        if opt.apex:
            with amp.scale_loss(loss,optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        t1 = time.time()
        FPS = opt.batchsize / (t1 - t0)
        print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Timer: {:.4f} sec. || FPS:{:.4f}".format(epoch, iteration, len(train_loader), loss.item(), (t1 - t0),FPS))
        #print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Timer: {:.4f} sec.".format(epoch, iteration, len(train_loader), loss.item(), (t1 - t0)))

def checkpoint(rrn, epoch): 
    save_model_path = os.path.join(opt.save_model_path, systime)
    isExists = os.path.exists(save_model_path)
    if not isExists:
        os.makedirs(save_model_path)
    model_name  = 'X'+str(opt.scale)+'_{}L'.format(opt.layer)+'_{}'.format(opt.patch_size)+'_epoch_{}.pth'.format(epoch)
    torch.save(rrn.state_dict(), os.path.join(save_model_path, model_name))
    print('Checkpoint saved to {}'.format(os.path.join(save_model_path, model_name)))

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.npu.manual_seed_all(seed)

if __name__ == '__main__':
    main()    
