# Copyright (c) Soumith Chintala 2016,
# All rights reserved
#
# Copyright 2020 Huawei Technologies Co., Ltd
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
# limitations under the License.

import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from senet import se_resnext50_32x4d

import math
from apex import amp
import apex
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# dataset setting
parser.add_argument('--data_path', metavar='DIR', default='/dataset/imagenet',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

# training setting
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--gamma', default=0.1, type=float,
                    metavar='GAM', help='decay factor of learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')

# distributed setting
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--device', default='npu', type=str, help='npu or gpu')
parser.add_argument('--device-list', default='0,1,2,3,4,5,6,7', type=str, help='device id list')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--addr', default='', type=str, help='master addr')
parser.add_argument('--port', default='', type=str, help='master port')

# apex setting
parser.add_argument('--amp', default=False, action='store_true',
                    help='use amp to train the model')
parser.add_argument('--opt-level', default=None, type=str, help='apex optimize level')
parser.add_argument('--loss-scale-value', default='1024', type=int, help='static loss scale value')

# other setting
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', default=10, type=int, help='test interval')
parser.add_argument('--stop-step-num', default=None, type=int, help='after the stop-step, killing the training task')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--profile', default=0, type=int, help="profile flag")

best_acc1 = 0
cur_step = 0


def device_id_to_process_device_map(device_list):
    devices = device_list.split(",")
    devices = [int(x) for x in devices]
    devices.sort()

    process_device_map = dict()
    for process_id, device_id in enumerate(devices):
        process_device_map[process_id] = device_id

    return process_device_map


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True


def profiling(loader, model, loss_fun, optimizer, args):
    # switch to train mode
    model.train()

    if 'npu' in args.device:
        loc = 'npu:{}'.format(args.gpu)
    else:
        loc = 'cuda:{}'.format(args.gpu)

    def update(model, images, target, optimizer):
        output = model(images)
        loss = loss_fun(output, target)
        optimizer.zero_grad()
        if args.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        optimizer.step()

    for i, (images, target) in enumerate(loader):
        if 'npu' in args.device:
            target = target.to(torch.int32)

        if 'npu' in args.device or 'cuda' in args.device:
            images = images.to(loc, non_blocking=True)
            target = target.to(loc, non_blocking=True)

        if i < 5:
            update(model, images, target, optimizer)
        else:
            if args.device == 'npu':
                with torch.autograd.profiler.profile(use_npu=True) as prof:
                    update(model, images, target, optimizer)
            elif args.device == "cuda":
                with torch.autograd.profiler.profile(use_cuda=True) as prof:
                    update(model, images, target, optimizer)
            break

    prof.export_chrome_trace("output.prof")


def main():
    args = parser.parse_args()

    print("----------before process----------")
    print(args)

    # if dist init method="env", use this method
    os.environ['MASTER_ADDR'] = args.addr
    os.environ['MASTER_PORT'] = args.port

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    args.process_device_map = device_id_to_process_device_map(args.device_list)

    if 'npu' in args.device:
        import torch.npu

    if args.seed is not None:
        seed_everything(args.seed)
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    if args.distributed:
        ngpus_per_node = len(args.process_device_map)
    else:
        ngpus_per_node = 1

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1

    args.gpu = args.process_device_map[gpu]

    if args.gpu is not None:
        print("[npu id:", args.gpu, "]", "Use NPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu

        if args.device == 'npu':
            dist.init_process_group(backend=args.dist_backend,  # init_method=args.dist_url,
                                    world_size=args.world_size, rank=args.rank)
        else:
            dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                    world_size=args.world_size, rank=args.rank)

    if args.device == 'npu':
        loc = 'npu:{}'.format(args.gpu)
        torch.npu.set_device(loc)
    else:
        loc = 'cuda:{}'.format(args.gpu)
        torch.cuda.set_device(loc)

    args.batch_size = int(args.batch_size / ngpus_per_node)
    args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)

    print("[npu id:", args.gpu, "]", "----------after process----------")
    print("[npu id:", args.gpu, "]", args)

    if args.pretrained:
        num_classes = 1000
        model = se_resnext50_32x4d()
        pretrained_dict = torch.load("./model_best.pth.tar", map_location="cpu")["state_dict"]
        model.load_state_dict({k.replace('module.', '', 1): v for k, v in pretrained_dict.items()})
        # 删除最后一个全连接层的参数
        if 'last_linear.weight' in pretrained_dict:
            pretrained_dict.pop('last_linear.weight')
            pretrained_dict.pop('last_linear.bias')
        if 'module.last_linear.weight' in pretrained_dict:
            pretrained_dict.pop('module.last_linear.weight')
            pretrained_dict.pop('module.last_linear.bias')
        # 冻结参数
        for param in model.parameters():
            param.requires_gard = False
        # 重新定义最后一层
        model.last_linear = nn.Linear(2048, num_classes)
        model.load_state_dict(pretrained_dict, strict=False)
    else:
        num_classes = 1000
        model = se_resnext50_32x4d(num_classes=num_classes)
    model = model.to(loc)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    criterion = nn.CrossEntropyLoss().to(loc)

    if args.amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level, loss_scale=args.loss_scale_value)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], broadcast_buffers=False)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if args.amp:
                amp.load_state_dict(checkpoint['amp'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading code
    traindir = os.path.join(args.data_path, 'train')
    valdir = os.path.join(args.data_path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False, sampler=val_sampler,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    if args.evaluate:
        acc1, acc5 = validate(val_loader, model, criterion, args, ngpus_per_node)
        return

    if args.profile:
        profiling(train_loader, model, criterion, optimizer, args)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # if epoch !=0 and epoch % 30 == 0:
        #     adjust_learning_rate(optimizer, args)
        adjust_learning_rate(optimizer, epoch, args)
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, ngpus_per_node)

        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            # evaluate on validation set
            acc1, acc5 = validate(val_loader, model, criterion, args, ngpus_per_node)

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            if epoch == args.epochs - 1 and args.rank % ngpus_per_node == 0:
                print("the best top1 is ", best_acc1)

            # save checkpoint
            if not args.multiprocessing_distributed or \
                    (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                print("save epoch is ", epoch)
                if args.amp:
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'best_acc1': best_acc1,
                        'optimizer': optimizer.state_dict(),
                        'amp': amp.state_dict(),
                    }, is_best, epoch)
                else:
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'best_acc1': best_acc1,
                        'optimizer': optimizer.state_dict(),
                    }, is_best, epoch)

        if args.stop_step_num is not None and cur_step >= args.stop_step_num:
            break


def train(train_loader, model, criterion, optimizer, epoch, args, ngpus_per_node):
    batch_time = AverageMeter('Time', ':6.3f', start_count_index=5)
    data_time = AverageMeter('Data', ':6.3f', start_count_index=5)
    losses = AverageMeter('Loss', ':6.8f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    if 'npu' in args.device:
        loc = 'npu:{}'.format(args.gpu)
    else:
        loc = 'cuda:{}'.format(args.gpu)

    end = time.time()
    steps_per_epoch = len(train_loader)
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if 'npu' in args.device:
            target = target.to(torch.int32)

        if 'npu' in args.device or 'cuda' in args.device:
            images = images.to(loc, non_blocking=True)
            target = target.to(loc, non_blocking=True)

        if 'npu' in args.device:
            stream = torch.npu.current_stream()
        else:
            stream = torch.cuda.current_stream()

        # compute output
        output = model(images)
        stream.synchronize()

        loss = criterion(output, target)
        stream.synchronize()

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, args.device, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        stream.synchronize()
        if args.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        stream.synchronize()
        optimizer.step()
        stream.synchronize()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i == 4:
            batch_time.reset()
        if i % args.print_freq == 0:
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                        and args.rank % ngpus_per_node == 0):
                progress.display(i)

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                and args.rank % ngpus_per_node == 0):
        print("[npu id:", args.gpu, "]", '* FPS@all {:.3f}'.format(ngpus_per_node * args.batch_size / batch_time.avg))


def validate(val_loader, model, criterion, args, ngpus_per_node):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    if 'npu' in args.device:
        loc = 'npu:{}'.format(args.gpu)
    else:
        loc = 'cuda:{}'.format(args.gpu)

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):

            if 'npu' in args.device:
                target = target.to(torch.int32)

            if 'npu' in args.device or 'cuda' in args.device:
                images = images.to(loc, non_blocking=True)
                target = target.to(loc, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, args.device, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                if not args.multiprocessing_distributed or \
                        (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    progress.display(i)

        if not args.multiprocessing_distributed or \
                (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            print("[npu id:", args.gpu, "]", '[AVG-ACC] * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


def save_checkpoint(state, is_best, epoch, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        print("======save best", " epoch ", epoch, "=======")
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', start_count_index=0):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.start_count_index = start_count_index

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if self.count == 0:
            self.N = n

        self.val = val
        self.count += n

        if self.count > (self.start_count_index * self.N):
            self.sum += val * n
            self.avg = self.sum / (self.count - self.start_count_index * self.N)

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        # 日志打点
        train_acc1 = str(entries).split("Acc@1")[1].strip().split(" ")[0]
        train_acc5 = str(entries).split("Acc@5")[1].strip().split(" ")[0]

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


# def adjust_learning_rate(optimizer, args):
#     """warm up cosine annealing learning rate."""
#     lr = args.lr * args.gamma
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

def adjust_learning_rate(optimizer, epoch, args):
    alpha = 0
    cosine_decay = 0.5 * (
            1 + np.cos((np.pi * epoch) / args.epochs))
    decayed = (1 - alpha) * cosine_decay + alpha
    lr = args.lr * decayed

    print("=> Now Epoch[%d] setting lr: %.4f" % (epoch, lr))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def accuracy(output, target, device, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # gpu use reshape, npu use view
            if device == "cuda":
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            elif device == "npu":
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    # print("ok")
    main()
