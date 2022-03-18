# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#coding=utf-8

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import time
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from apex.parallel import DistributedDataParallel as DDP
from apex import amp
import apex

from data.config import cur_config as cfg
from layers.modules import MultiBoxLoss
from data.widerface import WIDERDetection, detection_collate
from models.factory import build_net, basenet_factory

parser = argparse.ArgumentParser(
    description='DSFD face Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--batch_size',
                    default=16, type=int,
                    help='Batch size for training')
parser.add_argument('--model',
                    default='resnet152', type=str,
                    choices=['vgg', 'resnet50', 'resnet101', 'resnet152'],
                    help='model for training')
parser.add_argument('--resume',
                    default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--num_workers',
                    default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--npu',
                    default=True, type=bool,
                    help='Use npu to train model')
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
parser.add_argument("--dist_url", help="", default='127.0.0.1:6667', type=str)
parser.add_argument('--nodes', default=1, type=int, metavar='N')
parser.add_argument('--nr', default=0, type=int, help='ranking within the nodes')
parser.add_argument('--npus', default=1, type=int,help='number of gpus per node')
parser.add_argument('--device_id', default=0, type=int)

parser.add_argument('--save_folder',
                    default='weights/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--pretrain_weight',
                    default='./pretrain_weights/',
                    help='Directory for pretrained checkpoint models')

args = parser.parse_args()
args.is_master_node = not args.multinpu or args.device_id == 0
save_folder = os.path.join(args.save_folder, args.model)
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

train_dataset = WIDERDetection(cfg.FACE_TRAIN_FILE, mode='train')
val_dataset = WIDERDetection(cfg.FACE_VAL_FILE, mode='val')

val_batchsize = args.batch_size // 2
val_loader = data.DataLoader(val_dataset, val_batchsize,
                             num_workers=args.num_workers,
                             shuffle=False,
                             collate_fn=detection_collate,
                             pin_memory=True)
min_loss = np.inf

def train(args):
    per_epoch_size = len(train_dataset) // args.batch_size
    start_epoch = 0
    iteration = 0
    step_index = 0
    args.device = torch.device(f'npu:{args.device_id}')
    torch.npu.set_device(args.device)
    if args.multinpu:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29688'
        args.world_size = args.npus * args.nodes
        torch.distributed.init_process_group(backend='hccl',
                                             init_method='env://',
                                             world_size=args.world_size,
                                             rank=args.device_id)

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=args.world_size,
                                                                        rank=args.device_id)
        train_loader = data.DataLoader(train_dataset,
                                       batch_size=args.batch_size,
                                       shuffle=False,
                                       pin_memory=False,
                                       num_workers=8,
                                       sampler=train_sampler,
                                       collate_fn=detection_collate,
                                       drop_last=True)
    else:
        train_loader = data.DataLoader(train_dataset, args.batch_size,
                                       num_workers=args.num_workers,
                                       shuffle=True,
                                       collate_fn=detection_collate,
                                       pin_memory=True)

    basenet = basenet_factory(args.model)
    dsfd_net = build_net('train', cfg.NUM_CLASSES, args.model)
    net = dsfd_net

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        start_epoch = net.load_weights(args.resume)
        iteration = start_epoch * per_epoch_size
    else:
        base_weights = torch.load(args.pretrain_weight + basenet)
        print('Load base network {}'.format(args.pretrain_weight + basenet))
        if args.model == 'vgg':
            net.vgg.load_state_dict(base_weights)
        else:
            pretrained_dict = base_weights
            model_dict = net.resnet.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            net.resnet.load_state_dict(model_dict)

    criterion = MultiBoxLoss(cfg, args.npu, args.device)
    print(args)

    net = net.to(args.device)

    if not args.resume:
        print('Initializing weights...')
        dsfd_net.extras.apply(dsfd_net.weights_init)
        dsfd_net.fpn_topdown.apply(dsfd_net.weights_init)
        dsfd_net.fpn_latlayer.apply(dsfd_net.weights_init)
        dsfd_net.fpn_fem.apply(dsfd_net.weights_init)
        dsfd_net.loc_pal1.apply(dsfd_net.weights_init)
        dsfd_net.conf_pal1.apply(dsfd_net.weights_init)
        dsfd_net.loc_pal2.apply(dsfd_net.weights_init)
        dsfd_net.conf_pal2.apply(dsfd_net.weights_init)

    optimizer = torch.optim.SGD(net.parameters(),
                                            lr=args.lr,
                                            momentum=args.momentum,
                                            weight_decay=args.weight_decay)

    if args.multinpu:
        net, optimizer = amp.initialize(net, optimizer, opt_level="O2", loss_scale=16.0)
    else:
        net, optimizer = amp.initialize(net, optimizer, opt_level="O2", loss_scale=128.0)

    if args.npu:
        if args.multinpu:
            net = torch.nn.parallel.DistributedDataParallel(net, device_ids = [args.device_id])

    for step in cfg.LR_STEPS:
        if iteration > step:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

    net.train()
    for epoch in range(start_epoch, cfg.EPOCHES):
        losses = 0
        if args.multinpu:
            train_loader.sampler.set_epoch(epoch)
        for batch_idx, (images, targets) in enumerate(train_loader):
            if args.npu:
                images = Variable(images.to(args.device))
                targets = [Variable(ann.to(args.device), volatile=True)
                           for ann in targets]
            else:
                images = Variable(images)
                targets = [Variable(ann, volatile=True) for ann in targets]

            if iteration in cfg.LR_STEPS:
                step_index += 1
                adjust_learning_rate(optimizer, args.gamma, step_index)

            t0 = time.time()
            out = net(images)
            optimizer.zero_grad()
            loss_l_pa1l, loss_c_pal1 = criterion(out[:3], targets)
            loss_l_pa12, loss_c_pal2 = criterion(out[3:], targets)

            loss = loss_l_pa1l + loss_c_pal1 + loss_l_pa12 + loss_c_pal2
            #APEX
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            optimizer.step()
            t1 = time.time()
            losses += loss.data.item()
            if iteration % 10 == 0:
                tloss = losses / (batch_idx + 1)
                fps = args.batch_size / (t1 - t0)
                if args.multinpu:
                    fps = fps * 8
                print('Timer: %.4f' % (t1 - t0) + ' FPS:%.2f' % fps)
                print('epoch:' + repr(epoch) + ' || iter:' + repr(iteration) + ' || Loss:%.4f' % (tloss))
                print('->> pal1 conf loss:{:.4f} || pal1 loc loss:{:.4f}'.format(
                    loss_c_pal1.data.item(), loss_l_pa1l.data.item()))
                print('->> pal2 conf loss:{:.4f} || pal2 loc loss:{:.4f}'.format(
                    loss_c_pal2.data.item(), loss_l_pa12.data.item()))
                print('->>lr:{}'.format(optimizer.param_groups[0]['lr']))

            if iteration != 0 and iteration % 100 == 0:
                print('Saving state, iter:', iteration)
                file = 'dsfd_' + repr(iteration) + '.pth'
                if args.is_master_node:
                    torch.save(dsfd_net.state_dict(),os.path.join(save_folder, file))
                #torch.save(dsfd_net, os.path.join(save_folder, file)) #保存整个模型
            iteration += 1
        val(epoch, net, dsfd_net, criterion) #进行验证
        if iteration == cfg.MAX_STEPS:
            break


def val(epoch, net, dsfd_net, criterion):
    net.eval()
    step = 0
    losses = 0
    t1 = time.time()
    for batch_idx, (images, targets) in enumerate(val_loader):
        if args.npu:
            images = Variable(images.npu())
            targets = [Variable(ann.npu(), volatile=True)
                       for ann in targets]
        else:
            images = Variable(images)
            targets = [Variable(ann, volatile=True) for ann in targets]

        out = net(images)
        #loss_l_pa1l, loss_c_pal1 = criterion(out[:3], targets)
        loss_l_pa12, loss_c_pal2 = criterion(out[3:], targets)
        loss = loss_l_pa12 + loss_c_pal2
        losses += loss.data.item()
        step += 1

    tloss = losses / step
    t2 = time.time()
    print('Timer: %.4f' % (t2 - t1))
    print('test epoch:' + repr(epoch) + ' || Loss:%.4f' % (tloss))

    global min_loss
    if tloss < min_loss:
        print('Saving best state,epoch', epoch)
        torch.save(dsfd_net.state_dict(), os.path.join(save_folder, 'dsfd.pth'))
        min_loss = tloss

    states = {
        'epoch': epoch,
        'weight': dsfd_net.state_dict(),
    }
    torch.save(states, os.path.join(save_folder, 'dsfd_checkpoint.pth'))


def adjust_learning_rate(optimizer, gamma, step):
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    train(args)
