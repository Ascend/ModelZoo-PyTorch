#-*- coding:utf-8 -*-
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
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function


import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.utils.data as data

import os
import time
import torch
import argparse

import numpy as np
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.npu
from data.config import cfg
from pyramidbox import build_net
from layers.modules import MultiBoxLoss
from data.widerface import WIDERDetection, detection_collate
from torch.nn.parallel import DistributedDataParallel as DDP
import apex
from apex import amp
import torch.distributed as dist
import torch.multiprocessing as mp

parser = argparse.ArgumentParser(
    description='Pyramidbox face Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--basenet',
                    default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size',
                    default=16, type=int,
                    help='Batch size for training')
parser.add_argument('--resume',
                    default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--num_workers',
                    default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--npu',
                    default=True, type=bool,
                    help='Use NPU to train model')
parser.add_argument('--performance',
                    default=False, type=bool,
                    help='performance to train')
parser.add_argument('--lr', '--learning-rate',
                    default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum',
                    default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay',
                    default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma',
                    default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--multinpu',
                    default=False, type=bool,
                    help='Use mutil Gpu training')
parser.add_argument('--save_folder',
                    default='weights/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--local_rank',
                    default=-1, type=int,
                    help='rank for current process')
parser.add_argument('--world_size', default=-1, type=int,
                    help='number of distributed processes')
parser.add_argument('--device_list', default='0', type=str,
                    help='NPU id to use.')
args = parser.parse_args()

if args.npu:
  if args.multinpu:
        device_id = int(args.device_list.split(',')[args.local_rank])
        device = 'npu:{}'.format(device_id)
  else:
        device = 'npu:0'
  torch.npu.set_device(device)

if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder)

train_dataset = WIDERDetection(cfg.FACE.TRAIN_FILE, mode='train')

val_dataset = WIDERDetection(cfg.FACE.VAL_FILE, mode='val')
val_batchsize = 1
val_loader = data.DataLoader(val_dataset, val_batchsize,
                             num_workers=1,
                             shuffle=False,
                             collate_fn=detection_collate,
                             pin_memory=True)

min_loss = np.inf
def train():
    # torch.set_num_threads(1)
    iteration = 0
    start_epoch = 0
    step_index = 0
    per_epoch_size = len(train_dataset) // args.batch_size
    if args.local_rank==0 or args.multinpu==False:
      print('------build_net start-------')
    pyramidbox = build_net('train', cfg.NUM_CLASSES)
    if args.local_rank==0 or args.multinpu==False:
      print('------build_net end-------')
    net = pyramidbox
    if args.multinpu:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.num_workers,
            pin_memory=False,
            sampler=train_sampler,
            collate_fn=detection_collate,
            drop_last=True)
    else:
        train_loader = data.DataLoader(train_dataset, args.batch_size,
                               num_workers=args.num_workers,
                               shuffle=False,
                               collate_fn=detection_collate,
                               pin_memory=True)
    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        start_epoch = net.load_weights(args.resume)
        iteration = start_epoch * per_epoch_size
    else:
        vgg_weights = torch.load(args.save_folder + args.basenet)
        if args.local_rank==0 or args.multinpu==False:
            print('Load base network....')
        net.vgg.load_state_dict(vgg_weights)
    
    if args.local_rank==0 or args.multinpu==False:
        print('load base network end--------')
    if not args.resume:
        if args.local_rank==0 or args.multinpu==False:
          print('Initializing weights...')
        pyramidbox.bn64.apply(pyramidbox.weights_init)       
        pyramidbox.extras.apply(pyramidbox.weights_init)
        pyramidbox.lfpn_topdown.apply(pyramidbox.weights_init)
        pyramidbox.lfpn_later.apply(pyramidbox.weights_init)
        pyramidbox.cpm.apply(pyramidbox.weights_init)
        pyramidbox.loc_layers.apply(pyramidbox.weights_init)
        pyramidbox.conf_layers.apply(pyramidbox.weights_init)
    
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    
    if args.npu:
        net.npu()
        net, optimizer = amp.initialize(net, optimizer, opt_level="O1",loss_scale=64.0)#,combine_grad=True)
        if args.multinpu:
            device_id = int(args.device_list.split(',')[args.local_rank])
            device = 'npu:{}'.format(device_id)
            net = DDP(net, device_ids=[device_id],broadcast_buffers=False)
    cudnn.benckmark = True
    criterion1 = MultiBoxLoss(cfg, args.npu)
    criterion2 = MultiBoxLoss(cfg, args.npu, use_head_loss=True)
    if args.local_rank==0 or args.multinpu==False:
      print('Loading wider dataset...')
      print('Using the specified args:')
      print(args)  
    warmup_steps = 1000
    net.train()
    if args.local_rank==0 or args.multinpu==False:
      print('start train--------')
    for epoch in range(start_epoch, cfg.EPOCHES):
        if args.multinpu:
          train_sampler.set_epoch(epoch) 
        losses = 0
        for batch_idx, (images, face_targets, head_targets) in enumerate(train_loader):
            
            if args.npu:                
                images = Variable(images.npu())
                with torch.no_grad():
                    face_targets = [Variable(ann) for ann in face_targets]
                    head_targets = [Variable(ann) for ann in head_targets]
            else:
                images = Variable(images)
                with torch.no_grad():
                  face_targets = [Variable(ann) for ann in face_targets]
                  head_targets = [Variable(ann) for ann in head_targets]
            adjust_learning_rate(optimizer,iteration,warmup_steps,15000)
            t0 = time.time()            
            out = net(images)            
            optimizer.zero_grad()       
            face_loss_l, face_loss_c = criterion1(out, face_targets)
            head_loss_l, head_loss_c = criterion2(out, head_targets)
            loss = face_loss_l + face_loss_c + head_loss_l + head_loss_c
            losses += loss.item()
            if args.npu:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()            
            t1 = time.time()
            face_loss = (face_loss_l + face_loss_c).item()
            head_loss = (head_loss_l + head_loss_c).item()
            
            if args.performance:
                if iteration == 50:
                    t50_0 = time.time()
                if iteration == 100:
                    t100_0 = time.time()
                    if args.multinpu:
                      if args.local_rank==0:
                          print('cost time:{}  batch_size:{} num_gpu:{} FPS:{}'.format(t100_0-t50_0,args.batch_size,args.world_size,(50*args.batch_size*args.world_size)/(t100_0-t50_0)))
                    else:
                      print('cost time:{}  batch_size:{}  FPS:{}'.format(t100_0-t50_0,args.batch_size,(50*args.batch_size)/(t100_0-t50_0)))
                if iteration == 110:
                    break
                iteration += 1
                continue
            if iteration % 10 == 0 and (args.local_rank==0 or args.multinpu==False):
                loss_ = losses / (batch_idx + 1)
                print('Timer: {:.4f} sec.'.format(t1 - t0))
                print('epoch ' + repr(epoch) + ' iter ' +
                      repr(iteration) + ' || Loss:%.4f' % (loss_))
                print('->> face Loss: {:.4f} || head loss : {:.4f}'.format(
                    face_loss, head_loss))
                print('->> lr: {}'.format(optimizer.param_groups[0]['lr']))
                if args.multinpu:
                  print('iter:{}  cost time:{}  batch_size:{} num_gpu:{} FPS:{}'.format(iteration,t1-t0,args.batch_size,args.world_size,(args.batch_size*args.world_size)/(t1-t0)))
                else:
                  print('iter:{}  cost time:{}  batch_size:{}  FPS:{}'.format(iteration,t1-t0,args.batch_size,args.batch_size/(t1-t0)))
            if iteration != 0 and iteration % 2000 == 0 and (args.local_rank==0 or args.multinpu==False):
                print('Saving state, iter:', iteration)
                file = 'pyramidbox_' + repr(iteration) + '.pth'
                torch.save(pyramidbox.state_dict(),
                           os.path.join(args.save_folder, file))
            iteration += 1
        if args.performance:
            break
        if epoch>50 and epoch%5==0:
          val(epoch, net, pyramidbox, criterion1, criterion2)
          net.train()

def val(epoch,
        net,
        pyramidbox,
        criterion1,
        criterion2):
    net.eval()
    face_losses = 0
    head_losses = 0
    step = 0
    t1 = time.time()
    for batch_idx, (images, face_targets, head_targets) in enumerate(val_loader):
        if args.npu:
            images = Variable(images.npu())
            with torch.no_grad():
                    face_targets = [Variable(ann) for ann in face_targets]
                    head_targets = [Variable(ann) for ann in head_targets]
            
        else:
            images = Variable(images)
            with torch.no_grad():
              face_targets = [Variable(ann)
                            for ann in face_targets]
              head_targets = [Variable(ann)
                            for ann in head_targets]  

        out = net(images)
        face_loss_l, face_loss_c = criterion1(out, face_targets)
        head_loss_l, head_loss_c = criterion2(out, head_targets)

        face_losses += (face_loss_l + face_loss_c).item()
        head_losses += (head_loss_l + head_loss_c).item()
        step += 1

    tloss = face_losses / step

    t2 = time.time()
    if args.local_rank==0:
      print('test Timer:{:.4f} .sec'.format(t2 - t1))
      print('epoch ' + repr(epoch) + ' || Loss:%.4f' % (tloss))

    global min_loss
    if tloss < min_loss and args.local_rank==0:
        print('Saving best state,epoch', epoch)
        torch.save(pyramidbox.state_dict(), os.path.join(
            args.save_folder, 'pyramidbox.pth'))
        min_loss = tloss

    states = {
        'epoch': epoch,
        'weight': pyramidbox.state_dict(),
    }
    if args.local_rank==0:
        torch.save(states, os.path.join(
        args.save_folder, 'pyramidbox_checkpoint.pth'))
    

def lr_warmup(optimizer,step,base_lr,warmup_steps):
    if not step <warmup_steps:
        return
    lr = base_lr*(step+1)/warmup_steps
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate(optimizer,step,warmup_step,total_step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    #lr = args.lr * (gamma ** (step))
    lr = 0.5 * (1 + np.cos(np.pi * (step - warmup_step) / (total_step - warmup_step))) * args.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    if args.multinpu:
      print('in multi--------')
      dist.init_process_group(backend='hccl', world_size=args.world_size, rank=args.local_rank)
      train()
    
    else:
      print('in train------')
      train()
    
