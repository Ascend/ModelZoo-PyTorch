#!/usr/bin/env python
# -*- encoding: utf-8 -*-
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
'''
@File    :   main.py
@Time    :   2020/03/09
@Author  :   jhhuang96
@Mail    :   hjh096@126.com
@Version :   1.0
@Description:   
'''

import os
import torch.npu
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')
    print("========device_id:",NPU_CALCULATE_DEVICE)
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from encoder import Encoder
from decoder import Decoder
from model import ED
from net_params import convlstm_encoder_params, convlstm_decoder_params, convgru_encoder_params, convgru_decoder_params
from data.mm import MovingMNIST
import torch
from torch import nn
from torch.optim import lr_scheduler
import torch.optim as optim
import sys
from earlystopping import EarlyStopping
from tqdm import tqdm
import numpy as np
#from tensorboardX import SummaryWriter
import argparse
import time
try:
    from apex import amp
except:
    amp = None

TIMESTAMP = "2020-03-09T00-00-00"
parser = argparse.ArgumentParser()
parser.add_argument('-clstm',
                    '--convlstm',
                    help='use convlstm as base cell',
                    action='store_true')
parser.add_argument('-cgru',
                    '--convgru',
                    help='use convgru as base cell',
                    action='store_true')
parser.add_argument('--batch_size',
                    default=4,
                    type=int,
                    help='mini-batch size')
parser.add_argument('-lr', default=1e-4, type=float, help='G learning rate')
parser.add_argument('-frames_input',
                    default=10,
                    type=int,
                    help='sum of input frames')
parser.add_argument('-frames_output',
                    default=10,
                    type=int,
                    help='sum of predict frames')
parser.add_argument('-epochs', default=500, type=int, help='sum of epochs')
parser.add_argument('--apex', action='store_true',
                                       help='User apex for mixed precision training')
parser.add_argument('--apex-opt-level', default='O1', type=str,
                                       help='For apex mixed precision training'
                                                  'O0 for FP32 training, O1 for mixed precison training.')
parser.add_argument('--max_steps', default=None, type=int, metavar='N',
                        help='number of total steps to run')
args = parser.parse_args()

random_seed = 1996
np.random.seed(random_seed)
torch.manual_seed(random_seed)
#if torch.cuda.device_count() > 1:
if torch.npu.device_count() > 1:
    #torch.cuda.manual_seed_all(random_seed)
    torch.npu.manual_seed_all(random_seed)
else:
    #torch.cuda.manual_seed(random_seed)
    torch.npu.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

save_dir = './save_model/' + TIMESTAMP

trainFolder = MovingMNIST(is_train=True,
                          root='data/',
                          n_frames_input=args.frames_input,
                          n_frames_output=args.frames_output,
                          num_objects=[3])
validFolder = MovingMNIST(is_train=False,
                          root='data/',
                          n_frames_input=args.frames_input,
                          n_frames_output=args.frames_output,
                          num_objects=[3])
trainLoader = torch.utils.data.DataLoader(trainFolder,
                                          batch_size=args.batch_size,
                                          num_workers=96,
                                          shuffle=False)
validLoader = torch.utils.data.DataLoader(validFolder,
                                          batch_size=args.batch_size,
                                          num_workers=96,
                                          shuffle=False)

if args.convlstm:
    encoder_params = convlstm_encoder_params
    decoder_params = convlstm_decoder_params
if args.convgru:
    encoder_params = convgru_encoder_params
    decoder_params = convgru_decoder_params
else:
    encoder_params = convgru_encoder_params
    decoder_params = convgru_decoder_params


def train():
    '''
    main function to run the training
    '''
    #encoder = Encoder(encoder_params[0], encoder_params[1]).cuda()
    encoder = Encoder(encoder_params[0], encoder_params[1]).npu()
    #decoder = Decoder(decoder_params[0], decoder_params[1]).cuda()
    decoder = Decoder(decoder_params[0], decoder_params[1]).npu()
    net = ED(encoder, decoder)
    run_dir = './runs/' + TIMESTAMP
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)
    #tb = SummaryWriter(run_dir)
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=20, verbose=True)
    device = torch.device(f'npu:{NPU_CALCULATE_DEVICE}')

    #if torch.cuda.device_count() > 1:
    if torch.npu.device_count() > 1:
        #net = nn.DataParallel(net)
        net = net
    #net.to(device)
    net.to(f'npu:{NPU_CALCULATE_DEVICE}')

    if os.path.exists(os.path.join(save_dir, 'checkpoint.pth.tar')):
        # load existing model
        print('==> loading existing model')
        model_info = torch.load(os.path.join(save_dir, 'checkpoin.pth.tar'))
        net.load_state_dict(model_info['state_dict'])
        optimizer = torch.optim.Adam(net.parameters())
        optimizer.load_state_dict(model_info['optimizer'])
        cur_epoch = model_info['epoch'] + 1
    else:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        cur_epoch = 0
    #lossfunction = nn.MSELoss().cuda()
    lossfunction = nn.MSELoss().npu()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    pla_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                      factor=0.5,
                                                      patience=4,
                                                      verbose=True)
    if args.apex:
        net, optimizer = amp.initialize(net, optimizer,
                                                       opt_level=args.apex_opt_level, combine_grad=True)
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    # mini_val_loss = np.inf
    for epoch in range(cur_epoch, args.epochs + 1):
        ###################
        # train the model #
        ###################
        t = tqdm(trainLoader, leave=False, total=len(trainLoader))
        n = 0
        for i, (idx, targetVar, inputVar, _, _) in enumerate(t):
            if args.max_steps and i > args.max_steps:
                break
            start_time = time.time()
            inputs = inputVar.to(f'npu:{NPU_CALCULATE_DEVICE}')  # B,S,C,H,W
            label = targetVar.to(f'npu:{NPU_CALCULATE_DEVICE}')  # B,S,C,H,W
            optimizer.zero_grad()
            net.train()
            pred = net(inputs)  # B,S,C,H,W
            loss = lossfunction(pred, label)
            loss_aver = loss.item() / args.batch_size
            train_losses.append(loss_aver)
            if args.apex:
                with amp.scale_loss(loss,optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=10.0)
            optimizer.step()
            step_time = time.time() - start_time
            FPS = args.batch_size / step_time
            
            t.set_postfix({
                'step': '{}'.format(i),
                'trainloss': '{:.6f}'.format(loss_aver),
                'epoch': '{:02d}'.format(epoch),
                'time/step(s)': '{:.4f}'.format(step_time),
                'FPS': '{:.4f}'.format(FPS)
            })
        #tb.add_scalar('TrainLoss', loss_aver, epoch)
        ######################
        # validate the model #
        ######################
        with torch.no_grad():
            net.eval()
            t = tqdm(validLoader, leave=False, total=len(validLoader))
            for i, (idx, targetVar, inputVar, _, _) in enumerate(t):
                if i == 3000:
                    break
                inputs = inputVar.to(f'npu:{NPU_CALCULATE_DEVICE}')
                label = targetVar.to(f'npu:{NPU_CALCULATE_DEVICE}')
                pred = net(inputs)
                loss = lossfunction(pred, label)
                loss_aver = loss.item() / args.batch_size
                # record validation loss
                valid_losses.append(loss_aver)
                #print ("validloss: {:.6f},  epoch : {:02d}".format(loss_aver,epoch),end = '\r', flush=True)
                t.set_postfix({
                    'validloss': '{:.6f}'.format(loss_aver),
                    'epoch': '{:02d}'.format(epoch)
                })

        #tb.add_scalar('ValidLoss', loss_aver, epoch)
        #torch.cuda.empty_cache()
        torch.npu.empty_cache()
        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(args.epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{args.epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.6f} ' +
                     f'valid_loss: {valid_loss:.6f}')

        print(print_msg)
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        pla_lr_scheduler.step(valid_loss)  # lr_scheduler
        model_dict = {
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        early_stopping(valid_loss.item(), model_dict, epoch, save_dir)
        #if early_stopping.early_stop:
        #    print("Early stopping")
        #    break

    with open("avg_train_losses.txt", 'wt') as f:
        for i in avg_train_losses:
            print(i, file=f)

    with open("avg_valid_losses.txt", 'wt') as f:
        for i in avg_valid_losses:
            print(i, file=f)


if __name__ == "__main__":
    train()
