#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# coding=utf-8
from models.timeAcc import AverageMeter
from data import VOC_CLASSES as labelmap
from data import VOCAnnotationTransform, VOCDetection, BaseTransform
from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import RefineDetMultiBoxLoss
from models.refinedet import build_refinedet
from apex import amp
import apex
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
from utils.logging import Logger
from torch.hub import load_state_dict_from_url
import torch.npu
CALCULATE_DEVICE = "npu"

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--input_size', default='320', choices=['320', '512'],
                    type=str, help='RefineDet320 or RefineDet512')
parser.add_argument('--dataset_root', default='/home/ljh/refinedet/data/VOCdevkit/',
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='weights/vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_epoch', default=0, type=int,
                    help='Resume training at this epoch')
parser.add_argument('--num_epochs', default=232, type=int,
                    help='Total train epoch')     
parser.add_argument('--num_workers', default=14, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=False, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--npu', default=True, type=str2bool,
                    help='Use NPU to train model')
parser.add_argument('--lr', '--learning-rate', default=0.00095, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--finetune', default=None, type=str,
                    help='pretrained weight path')
parser.add_argument('--train_1p', default=False, type=str2bool,
                    help='node rank for distributed training')
parser.add_argument('--device_id', default=0, type=str,
                    help='device_id')
parser.add_argument('--amp', default=True, type=str2bool,
                    help='whether to use amp')
parser.add_argument('--num_classes', default=-1, type=int,
                    help='num classes')
args = parser.parse_args()


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

sys.stdout = Logger(os.path.join(args.save_folder, 'log.txt'))

def train():
    torch.npu.set_device('npu:' + str(args.device_id))
    
    if args.dataset == 'VOC':
        '''if args.dataset_root == COCO_ROOT:
            parser.error('Must specify dataset if specifying dataset_root')'''
        cfg = voc_refinedet[args.input_size]
        dataset = VOCDetection(root=args.dataset_root,
                               transform=SSDAugmentation(cfg['min_dim'], MEANS))  # cfg['min_dim'] = 320
    
    if args.finetune:
        print('finetune numclass %d'%args.num_classes)
        refinedet_net = build_refinedet('train', cfg['min_dim'], args.num_classes, batch_norm=True)
    else:
        refinedet_net = build_refinedet('train', cfg['min_dim'], cfg['num_classes'], batch_norm=True)
    net = refinedet_net
    if args.cuda:
        net = net.cuda()
    if args.npu:
        net = net.npu()
    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        refinedet_net.load_weights(args.resume)
    else:
        print('Loading vgg...')
        current_path = os.path.abspath(os.path.dirname(__file__))
        with open(os.path.join(current_path, './url.ini'), 'r') as _f:
            _content = _f.read()
            vgg16_pth_url = _content.split('vgg16_pth_url=')[1].split('\n')[0]
        vgg_weights = load_state_dict_from_url(vgg16_pth_url, progress=True)
        from collections import OrderedDict
        new_vgg_weights = OrderedDict()
        for k, v in vgg_weights.items():
            fc, num, wb = k.split('.')
            if fc == 'classifier':
                continue
            new_k = num + '.' + wb
            new_vgg_weights[new_k] = v
        refinedet_net.vgg.load_state_dict(new_vgg_weights, strict=False)
    if not args.resume:
        print('Initializing weights...')
        refinedet_net.extras.apply(weights_init)
        refinedet_net.arm_loc.apply(weights_init)
        refinedet_net.arm_conf.apply(weights_init)
        refinedet_net.odm_loc.apply(weights_init)
        refinedet_net.odm_conf.apply(weights_init)
        refinedet_net.tcb0.apply(weights_init)
        refinedet_net.tcb1.apply(weights_init)
        refinedet_net.tcb2.apply(weights_init)
    optimizer = apex.optimizers.NpuFusedSGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    arm_criterion = RefineDetMultiBoxLoss(2, 0.5, True, 0, True, 3, 0.5,
                                          False, args.cuda, npu_device=CALCULATE_DEVICE)
    if args.finetune:
        stat_dict = torch.load(args.finetune, map_location='cpu')
        for k in stat_dict.keys():
            if 'odm_conf' in k:
                stat_dict.pop(k)
        net.load_state_dict(stat_dict, strict=False)
        odm_criterion = RefineDetMultiBoxLoss(args.num_classes, 0.5, True, 0, True, 3, 0.5,
                                          False, args.cuda, use_ARM=True, npu_device=CALCULATE_DEVICE)
    else:
        odm_criterion = RefineDetMultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                                          False, args.cuda, use_ARM=True, npu_device=CALCULATE_DEVICE)
    if args.amp:
        net, optimizer = amp.initialize(net, optimizer, opt_level='O1', loss_scale=128, combine_grad=True)
    if args.cuda:
        net = torch.nn.DataParallel(refinedet_net)
        cudnn.benchmark = True
    net.train()
    arm_loc_loss = 0
    arm_conf_loss = 0
    odm_loc_loss = 0
    odm_conf_loss = 0
    print('Loading the dataset...')
    epoch_size = len(dataset) // args.batch_size
    if len(dataset) % args.batch_size != 0:
        epoch_size += 1
    print('Training RefineDet on:', dataset.name)
    print('Using the specified args:')
    print(args)
    step_index = 0
    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True,
                                  drop_last=True)

    if args.resume:
        strat_iter = args.start_epoch * epoch_size
        for step in cfg['lr_steps']:
            if strat_iter > step:
                step_index += 1
                adjust_learning_rate(optimizer, args.gamma, step_index)

    for epoch in range(args.start_epoch, args.num_epochs):
        avg_time = AverageMeter('iter_time')
        print('\n' + 'epoch ' + str(epoch))
        print('================================train model on trainval set================================')
        for iteration, (images, targets) in zip(range(epoch * epoch_size, (epoch + 1) * epoch_size), data_loader):
            if iteration in cfg['lr_steps']:
                step_index += 1
                adjust_learning_rate(optimizer, args.gamma, step_index)

            if args.cuda:
                images = images.cuda()
                targets = [ann.cuda() for ann in targets]
            elif args.npu:
                images = images.to(CALCULATE_DEVICE)
                targets = [ann.to(CALCULATE_DEVICE) for ann in targets]
            else:
                images = images
                targets = [ann for ann in targets]
            t0 = time.time()
            out = net(images)
            optimizer.zero_grad()
            arm_loss_l, arm_loss_c = arm_criterion(out, targets)
            odm_loss_l, odm_loss_c = odm_criterion(out, targets)
            arm_loss = arm_loss_l + arm_loss_c
            odm_loss = odm_loss_l + odm_loss_c
            loss = arm_loss + odm_loss
            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()
            t1 = time.time()
            arm_loc_loss += arm_loss_l.item()
            arm_conf_loss += arm_loss_c.item()
            odm_loc_loss += odm_loss_l.item()
            odm_conf_loss += odm_loss_c.item()
            avg_time.update(t1 - t0)
            if iteration % 10 == 0:
                print('iter ' + repr( \
                    iteration) + ' || ARM_L Loss: %.4f ARM_C Loss: %.4f ODM_L Loss: %.4f ODM_C Loss: %.4f ||' \
                      % (arm_loss_l.item(), arm_loss_c.item(), odm_loss_l.item(), odm_loss_c.item()), end=' ')
                print('timer: %.4f sec.' % (t1 - t0))

        print('batch_size = ' + str(args.batch_size) + ' || num_devices = ' + '1' + ' || time_avg = %.4f' % avg_time.avg)
        print('FPS = %.4f' % (args.batch_size / avg_time.avg))
        print('Saving state, iter:' + str(epoch_size * (epoch + 1) - 1) + ' , epoch:' + str(epoch))
        save_path = args.save_folder + '/RefineDet{}_{}_{}.pth'.format(args.input_size, args.dataset, epoch)
        torch.save(refinedet_net.state_dict(), save_path)

def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def xavier(param):
    init.xavier_uniform_(param)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        with torch.no_grad():
            init.xavier_uniform_(m.weight)
        with torch.no_grad():
            m.bias.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        xavier(m.weight.data)
        with torch.no_grad():
            m.bias.zero_()
if __name__ == '__main__':
    train()


