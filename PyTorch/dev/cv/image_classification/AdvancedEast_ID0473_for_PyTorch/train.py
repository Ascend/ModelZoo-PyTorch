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

import torch
if torch.__version__ >= "1.8":
    import torch_npu
from torch.utils import data
from torch import nn
from torch.optim import lr_scheduler
from generator import custom_dataset
from model import EAST
from loss import Loss
import os
import time
import numpy as np
import cfg
import argparse
# 使能混合精度
try:
    from apex import amp
except ImportError:
    amp = None
import apex
# 使能混合精度

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', default=5, type=int, help='device_id')
    parser.add_argument('--apex', action='store_true',
                        help='User apex for mixed precision training')
    parser.add_argument('--apex-opt-level', default='O1', type=str,
                        help='For apex mixed precision training'
                             'O0 for FP32 training, O1 for mixed precison training.')
    parser.add_argument('--loss-scale-value', default=1024., type=float,
                        help='loss scale using in amp, default -1 means dynamic')
    args = parser.parse_args()

    return args

def train(train_img_path, pths_path, batch_size, lr,decay, num_workers, epoch_iter, interval,pretained,args):
    file_num = len(os.listdir(train_img_path))
    trainset = custom_dataset(train_img_path)
    train_loader = data.DataLoader(trainset, batch_size=batch_size, \
            shuffle=True, num_workers=num_workers, drop_last=True)
    
    criterion = Loss()
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(f'npu:{args.device_id}' if torch.npu.is_available() else "cpu")
    torch.npu.set_device(device)
    model = EAST()
    # TODO
    if os.path.exists(pretained):
        model.load_state_dict(torch.load(pretained, map_location="cpu"))
        #model.load_state_dict(torch.load(pretained))
        
    data_parallel = False
    #if torch.cuda.device_count() > 1:

            
    model.to(device)
    # 性能调优，替换API
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=decay)
    optimizer = apex.optimizers.NpuFusedAdam(model.parameters(), lr=lr, weight_decay=decay)
    # 性能调优，替换API

    # scheduler = lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.94)
    if args.apex:
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=args.apex_opt_level,
                                          loss_scale=args.loss_scale_value,
                                          combine_grad=True)

    if torch.npu.device_count() > 1:
        model = nn.DataParallel(model)
        data_parallel = True
    for epoch in range(epoch_iter):
        model.train()
        # optimizer.step()
        epoch_loss = 0
        epoch_time = time.time()
        for i, (img, gt_map) in enumerate(train_loader):
            start_time = time.time()
            img, gt_map = img.to(device),gt_map.to(device)
            east_detect = model.module(img)
            loss = criterion(gt_map, east_detect)
                
            epoch_loss += loss.item()
            optimizer.zero_grad()
            if args.apex:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()
                
            print('Epoch is [{}/{}], mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}, fps is {:.2f}'.format(\
                    epoch+1, epoch_iter, i+1, int(file_num/batch_size), time.time()-start_time, loss.item(), batch_size/(time.time()-start_time)))
            
        print('epoch_loss is {:.8f}, epoch_time is {:.8f}'.format(epoch_loss/int(file_num/batch_size), time.time()-epoch_time))
        print(time.asctime(time.localtime(time.time())))
        print('='*50)
        if (epoch + 1) % interval == 0:
            state_dict = model.module.state_dict() if data_parallel else model.state_dict()
            torch.save(state_dict, os.path.join(pths_path, cfg.train_task_id+'_model_epoch_{}.pth'.format(epoch+1)))


# def test():


if __name__ == '__main__':
    train_img_path = os.path.join(cfg.data_dir,cfg.train_image_dir_name)
    pths_path      = './saved_model'
    batch_size     = 10
    lr             = 1e-3
    decay          =5e-4
    num_workers    = 4
    epoch_iter     = 600
    save_interval  = 5
    pretained = './saved_model/mb3_512_model_epoch_535.pth'
    args = parse_args()
    train(train_img_path, pths_path, batch_size, lr, decay,num_workers, epoch_iter, save_interval,pretained,args)
