# -*- coding: utf-8 -*-
# BSD 3-Clause License
#
# Copyright (c) 2017, 
# All rights reserved.
#
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://spdx.org/licenses/BSD-3-Clause.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License


import os
import warnings
import argparse
import time
import random
import torch
if torch.__version__ >= "1.8":
    import torch_npu
import apex
from apex import amp
import torch.nn as nn
import torch.distributed as dist
from network import ShuffleNetV1
from utils import accuracy, AvgrageMeter, CrossEntropyLabelSmooth, save_checkpoint, get_lastest_model, get_parameters
from utils import get_pytorch_train_loader, get_pytorch_val_loader, adjust_learning_rate
try:
    from torch_npu.utils.profiler import Profile
except:
    print("Profile not in torch_npu.utils.profiler now..Auto Profile disabled.", flush=True)
    class Profile:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def end(self):
            pass

warnings.filterwarnings('ignore')
best_acc1 = 0


def get_args():
    parser = argparse.ArgumentParser("ShuffleNetV1")
    parser.add_argument('--data', default='/opt/npu/dataset/imagenet',
	                    help='path to dataset')
    parser.add_argument('--batch-size', type=int, default=1024, 
	                    help='batch size')
    parser.add_argument('--local-rank', type=int, default=0, 
	                    help='local rank')
    parser.add_argument('--device-num', type=int, default=1, 
	                    help='device number')
    parser.add_argument('--rank', default=0, type=int, 
	                    help='node rank for distributed training')
    parser.add_argument('--learning-rate', type=float, default=1, 
	                    help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, 
	                    help='momentum')
    parser.add_argument('--weight-decay', type=float, default=4e-5, 
	                    help='weight decay')
    parser.add_argument('--master-node', default=False, action='store_true')
    parser.add_argument('--warm_up_epochs', default=0, type=int, 
	                    help='warm up')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true', 
	                    help='use pre-trained model')
    parser.add_argument('--pretrain_pth_path', default='', type=str, metavar='PATH', 
	                    help='path to checkpoint for pretrain (default: none)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', 
	                    help='path to latest checkpoint (default: none)')
    parser.add_argument('--workers', default=24, type=int, metavar='N', 
	                    help='number of data loading workers ')
    parser.add_argument('--epochs', default=240, type=int, metavar='N', 
	                    help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', 
	                    help='start epoch of total epochs to run')
    parser.add_argument('--seed', default=123456, type=int, 
	                    help='seed for initializing training.')
    parser.add_argument('--addr', default='127.0.0.1', type=str, 
	                    help='master addr')
    parser.add_argument('--display-interval', type=int, default=20, 
	                    help='display interval')
    parser.add_argument('--eval-freq', default=5, type=int, metavar='N', 
	                    help='evaluate frequency (default: 5)')
    parser.add_argument('--world-size', default=-1, type=int, 
	                    help='number of nodes for distributed training')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', 
	                    help='evaluate model on validation set, must also use --resume to specify trained model')
    parser.add_argument('--amp', action='store_true', 
	                    help='use amp to train the model')
    parser.add_argument('--loss-scale', type=float, default=None, 
	                    help='loss scale using in amp')
    parser.add_argument('--opt-level', type=str, default='O2', 
	                    help='loss scale using in amp')
    parser.add_argument('--group', type=int, default=3, 
	                    help='group number')
    parser.add_argument('--model-size', type=str, default='2.0x', choices=['0.5x', '1.0x', '1.5x', '2.0x'], 
	                    help='size of the model')

    args = parser.parse_args()
    return args


def main():
    global best_acc1
    args = get_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    warnings.warn('You have chosen to seed training. '
                  'This will turn on the CUDNN deterministic setting, '
                  'which can slow down your training considerably! '
                  'You may see unexpected behavior when restarting '
                  'from checkpoints.')
    print('Using NPU {}'.format(args.local_rank))
    device = torch.device('npu:{}'.format(args.local_rank))
    torch.npu.set_device(device)

    os.environ['MASTER_ADDR'] = args.addr
    os.environ['MASTER_PORT'] = '29501'

    if args.device_num > 1:
        args.distributed = True
        args.world_size = args.device_num * args.world_size
        args.rank = args.rank * args.device_num + args.local_rank
        dist.init_process_group(backend='hccl', world_size=args.world_size, rank=args.rank)
        if args.local_rank == 0:
            args.master_node = True
    else:
        args.distributed = False
        args.master_node = True
    print(args)
    use_npu = False
    if torch.npu.is_available():
        use_npu = True
    assert os.path.exists(args.data)
    args.batch_size = int(args.batch_size / args.world_size)
    # Data loading code
    train_loader, train_loader_len, train_sampler = get_pytorch_train_loader(args.data, args.batch_size,
                                                                       workers=args.workers,
                                                                       distributed=args.distributed)

    val_loader = get_pytorch_val_loader(args.data, args.batch_size, args.workers, distributed=False)
    print('load data successfully')
    # create model
    if args.pretrained:
        print("[npu id:", args.local_rank, "]", "=> creating model")       
        model = ShuffleNetV1(group=args.group, model_size=args.model_size)
        pretrained_dict = \
        torch.load(args.pretrain_pth_path, map_location="cpu")["state_dict"]
        model.load_state_dict({k.replace('module.',''): v for k, v in pretrained_dict.items()})
        model.load_state_dict(pretrained_dict, strict=False)        
    else:
        print("[npu id:", args.local_rank, "]", "=> creating model")
        model = ShuffleNetV1(group=args.group, model_size=args.model_size)
    model = model.to(device)
    optimizer = apex.optimizers.NpuFusedSGD(get_parameters(model),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    criterion_smooth = CrossEntropyLabelSmooth(1000, 0.1)
    # 动态loss scale无法触发精度溢出检测，原因是1980B精度溢出检测机制没有适配，目前先改成静态loss scale规避
    model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level, combine_grad=True,loss_scale=16) 
    loss_function = criterion_smooth.npu()
    if args.device_num > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False)
    args.optimizer = optimizer
    args.loss_function = loss_function
    args.train_loader = train_loader
    args.val_loader = val_loader
    model_loaded = False
    if args.resume and not args.pretrained:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=device)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if args.amp:
                amp.load_state_dict(checkpoint['amp'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            model_loaded = True
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    if args.evaluate:
        if not model_loaded:
            print('Error: please load the model you want to evaluate with --resume !')
            return
        validate(model, device, args, epoch=epoch)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)
        train(model, device, args, epoch=epoch)
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1 or epoch > int(args.epochs * 0.9):
            # evaluate on validation set
            acc1 = validate(model, device, args, epoch=epoch)
            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            if not args.distributed or \
                    (args.distributed and args.local_rank % args.device_num == 0):
                if args.amp:
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'best_acc1': best_acc1,
                        'optimizer': optimizer.state_dict(),
                        'amp': amp.state_dict(),
                    }, is_best)
                else:
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'best_acc1': best_acc1,
                        'optimizer': optimizer.state_dict(),
                    }, is_best)


def adjust_bn_momentum(model, iters):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 1 / iters

def train(model, device, args, epoch, bn_process=False):
    batch_time = AvgrageMeter()
    optimizer = args.optimizer
    loss_function = args.loss_function
    train_loader = args.train_loader
    t1 = time.time()
    Top1_err, Top5_err = 0.0, 0.0
    model.train()
    mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).view(1, 3, 1, 1)
    std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).view(1, 3, 1, 1)
    mean = mean.to(device, non_blocking=True)
    std = std.to(device, non_blocking=True)
    profile = Profile(start_step=int(os.getenv('PROFILE_START_STEP', 10)),
                      profile_type=os.getenv('PROFILE_TYPE'))

    for iters, (data, target) in enumerate(train_loader):
        if bn_process:
            adjust_bn_momentum(model, iters)
        torch.npu.synchronize()
        d_st = time.time()
        target = target.type(torch.LongTensor)
        data = data.to(device, non_blocking=True).to(torch.float).sub(mean).div(std)
        target = target.to(device, non_blocking=True)
        handle_data_time = time.time() - d_st
        profile.start()
        output = model(data)
        loss = loss_function(output, target)
        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
        profile.end()
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        torch.npu.synchronize()
        train_time = time.time() - d_st
        Top1_err += 1 - prec1.item() / 100
        Top5_err += 1 - prec5.item() / 100

        batch_time.update(time.time() - t1)
        if args.master_node:
            printInfo = 'Epoch {}: step = {},  loss = {:.6f},  '.format(epoch, iters, loss.item()) + \
                        'Top-1 err = {:.6f},\t'.format(Top1_err / args.display_interval) + \
                        'Top-5 err = {:.6f},\t'.format(Top5_err / args.display_interval) + \
                        'data_time = {:.6f},\ttrain_time = {:.6f}'.format(handle_data_time, train_time)
            print(printInfo)
        t1 = time.time()
        Top1_err, Top5_err = 0.0, 0.0

    if args.device_num == 1 or (args.local_rank == 0 and args.device_num > 1):
        print("[npu id:", args.local_rank, "]", '* FPS@all {:.3f}, TIME@all {:.3f}'.format(args.device_num * args.batch_size / batch_time.avg, batch_time.avg))


def validate(model, device, args, epoch=0):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    loss_function = args.loss_function
    val_loader = args.val_loader
    model.eval()
    mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).view(1, 3, 1, 1)
    std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).view(1, 3, 1, 1)
    mean = mean.to(device, non_blocking=True)
    std = std.to(device, non_blocking=True)
    t1  = time.time()
    with torch.no_grad():
        for iters, (data, target) in enumerate(val_loader):
            target = target.type(torch.LongTensor)
            data = data.to(device, non_blocking=True).to(torch.float).sub(mean).div(std)
            target = target.to(device, non_blocking=True)
            output = model(data)
            loss = loss_function(output, target)
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            n = data.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            logInfo = 'Test {}: step = {}, \tloss = {:.6f},\t'.format(epoch, iters, objs.avg) + \
                      'Top-1 err = {:.6f},\t'.format(1 - top1.avg / 100) + \
                      'Top-5 err = {:.6f},\t'.format(1 - top5.avg / 100) + \
                      'val_time = {:.6f}'.format(time.time() - t1)

            print(logInfo)
        print("[npu id:", args.local_rank, "]", '[AVG-ACC] * Acc@1 {top1.avg:.3f}, Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    return top1.avg


def load_checkpoint(net, checkpoint):
    from collections import OrderedDict
    temp = OrderedDict()
    if 'state_dict' in checkpoint:
        checkpoint = dict(checkpoint['state_dict'])
    for k in checkpoint:
        k2 = 'module.'+k if not k.startswith('module.') else k
        temp[k2] = checkpoint[k]
    net.load_state_dict(temp, strict=True)

if __name__ == "__main__":
    main()

