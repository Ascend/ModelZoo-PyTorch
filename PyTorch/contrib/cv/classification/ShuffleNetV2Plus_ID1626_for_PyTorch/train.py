# -*- coding: utf-8 -*-
# BSD 3-Clause License
#
# Copyright (c) 2017
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


import argparse
import os
import random
import shutil
import time
import warnings

import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import apex
from apex import amp
from network import ShuffleNetV2_Plus
from multi_epochs_dataloader import MultiEpochsDataLoader
from utils import get_parameters, LabelSmoothingCrossEntropy

BATCH_SIZE = 1024
OPTIMIZER_BATCH_SIZE = 2048

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default='/opt/npu/dataset/imagenet',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=360, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=BATCH_SIZE, type=int,
                    metavar='N',
                    help='mini-batch size (default: 1024), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.5, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=4e-5, type=float,
                    metavar='W', help='weight decay (default: 4e-5)',
                    dest='weight_decay')
parser.add_argument('--workspace', type=str, default='./', metavar='DIR',
                    help='path to directory where checkpoints will be stored')
parser.add_argument('-p', '--print-freq', default=5, type=int,
                    metavar='N', help='print frequency (default: 5)')
parser.add_argument('-ef', '--eval-freq', default=5, type=int,
                    metavar='N', help='evaluate frequency (default: 5)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set, must also use --resume to specify trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('-bm', '--benchmark', default=0, type=int,
                    metavar='N', help='set benchmark status (default: 0, does not run benchmark)')
parser.add_argument('--device', default='npu', type=str, help='npu or gpu')
parser.add_argument('--addr', default='10.136.181.115', type=str, help='master addr')
parser.add_argument('--checkpoint-nameprefix', default='checkpoint', type=str, help='checkpoint-nameprefix')
parser.add_argument('--checkpoint-freq', default=0, type=int,
                    metavar='N', help='checkpoint frequency (default: 0)'
                                      '0: save only one file whitch per epoch;'
                                      'n: save diff file per n epoch'
                                      '-1:no checkpoint,not support')
parser.add_argument('--device_num', default=-1, type=int,
                    help='device_num')
parser.add_argument('--device-list', default='', type=str, help='device id list')
parser.add_argument('--warm_up_epochs', default=0, type=int,
                    help='warm up')
parser.add_argument('--run-prof', default=False, action='store_true', help='run profiling')

# apex
parser.add_argument('--amp', default=False, action='store_true',
                    help='use amp to train the model')
parser.add_argument('--loss-scale', default=None, type=float,
                    help='loss scale using in amp')
parser.add_argument('--opt-level', default='O2', type=str,
                    help='loss scale using in amp')

parser.add_argument('--model-size', type=str, default='Small', choices=['Small', 'Medium', 'Large'], help='size of the model')

warnings.filterwarnings('ignore')
best_acc1 = 0


def main():
    args = parser.parse_args()
    print("===============main()=================")
    print(args)
    print("===============main()=================")

    os.environ['LOCAL_DEVICE_ID'] = str(0)
    print("+++++++++++++++++++++++++++LOCAL_DEVICE_ID:", os.environ['LOCAL_DEVICE_ID'])

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    os.environ['MASTER_ADDR'] = args.addr  # '10.136.181.51'
    os.environ['MASTER_PORT'] = '29688'

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if args.device_list != '':
        ngpus_per_node = len(args.device_list.split(','))
    elif args.device_num != -1:
        ngpus_per_node = args.device_num
    elif args.device == 'npu':
        ngpus_per_node = torch.npu.device_count()
    else:
        ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        # The child process uses the environment variables of the parent process,
        # we have to set LOCAL_DEVICE_ID for every proc
        if args.device == 'npu':
            # main_worker(args.gpu, ngpus_per_node, args)
            mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
        else:
            mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    if args.device_list != '':
        args.gpu = int(args.device_list.split(',')[gpu])
    else:
        args.gpu = gpu

    print("[npu id:", args.gpu, "]", "++++++++++++++++ before set LOCAL_DEVICE_ID:", os.environ['LOCAL_DEVICE_ID'])
    os.environ['LOCAL_DEVICE_ID'] = str(args.gpu)
    print("[npu id:", args.gpu, "]", "++++++++++++++++ LOCAL_DEVICE_ID:", os.environ['LOCAL_DEVICE_ID'])

    if args.gpu is not None:
        print("[npu id:", args.gpu, "]", "Use GPU: {} for training".format(args.gpu))

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

    loc = 'npu:{}'.format(args.gpu)
    torch.npu.set_device(loc)

    args.batch_size = int(args.batch_size / args.world_size)
    args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)

    print("[npu id:", args.gpu, "]", "===============main_worker()=================")
    print("[npu id:", args.gpu, "]", args)
    print("[npu id:", args.gpu, "]", "===============main_worker()=================")

    train_loader, train_loader_len, train_sampler = get_pytorch_train_loader(args.data,
                                                                             args.batch_size,
                                                                             workers=args.workers,
                                                                             distributed=args.distributed)

    val_loader = get_pytorch_val_loader(args.data, args.batch_size, args.workers, distributed=False)

    # create model
    print("[npu id:", args.gpu, "]", "=> creating model")
    architecture = [0, 0, 3, 1, 1, 1, 0, 0, 2, 0, 2, 1, 1, 0, 2, 0, 2, 1, 3, 2]
    model = ShuffleNetV2_Plus(architecture=architecture, model_size=args.model_size)
    model = model.to(loc)

    # define loss function (criterion) and optimizer
    criterion = LabelSmoothingCrossEntropy(1000, 0.1).to(loc)
    optimizer = apex.optimizers.NpuFusedSGD(get_parameters(model), lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level, loss_scale=args.loss_scale)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], broadcast_buffers=False)

    # optionally resume from a checkpoint
    model_loaded = False
    if args.resume:
        if os.path.isfile(args.resume):
            if args.pretrained:
                pretrained_dict = \
                torch.load(args.resume, map_location="cpu")["state_dict"]
                pretrained_dict.pop('module.fc.0.weight')
                pretrained_dict.pop('module.classifier.0.weight')
                model.load_state_dict(pretrained_dict, strict=False)
                print("=> loaded pretrained model '{}'".format(args.resume))
            else:
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume, map_location=loc)
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

    cudnn.benchmark = True

    if args.run_prof:
        runprof(train_loader, model, criterion, optimizer, args)
        return

    if args.evaluate:
        if not model_loaded:
            print('Error: please load the model you want to evaluate with --resume !')
            return
        validate(val_loader, model, criterion, args, ngpus_per_node)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, train_loader_len, model, criterion, optimizer, epoch, args, ngpus_per_node)

        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1 or epoch > int(args.epochs * 0.9):
            # evaluate on validation set
            acc1 = validate(val_loader, model, criterion, args, ngpus_per_node)

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            if not args.multiprocessing_distributed or \
                    (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
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


def runprof(train_loader, model, criterion, optimizer, args):
    loc = 'npu:{}'.format(args.gpu)

    mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).view(1, 3, 1, 1)
    std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).view(1, 3, 1, 1)
    mean = mean.to(loc, non_blocking=True)
    std = std.to(loc, non_blocking=True)

    model.train()
    cnt = 0
    for _, (images, target) in enumerate(train_loader):
        cnt += 1

        images = images.to(loc, non_blocking=True).to(torch.float).sub(mean).div(std)
        target = target.to(loc, non_blocking=True)
        # profile after 5 steps
        if cnt > 5:
            with torch.autograd.profiler.profile(use_npu=True) as prof:
                output = model(images)
                loss = criterion(output, target)
                optimizer.zero_grad()
                if args.amp:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                optimizer.step()
            prof.export_chrome_trace("output.prof")
            return
        output = model(images)
        loss = criterion(output, target)
        optimizer.zero_grad()
        if args.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()


def train(train_loader, train_loader_len, model, criterion, optimizer, epoch, args, ngpus_per_node):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e', start_count_index=0)
    top1 = AverageMeter('Acc@1', ':6.2f', start_count_index=0)
    top5 = AverageMeter('Acc@5', ':6.2f', start_count_index=0)
    progress = ProgressMeter(
        train_loader_len,
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    loc = 'npu:{}'.format(args.gpu)

    mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).view(1, 3, 1, 1)
    std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).view(1, 3, 1, 1)
    mean = mean.to(loc, non_blocking=True)
    std = std.to(loc, non_blocking=True)

    # switch to train mode
    model.train()
    end = time.time()
    if args.benchmark == 1:
        optimizer.zero_grad()

    steps_per_epoch = train_loader_len
    print('==========step per epoch======================', steps_per_epoch)
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.to(loc, non_blocking=True).to(torch.float).sub(mean).div(std)
        target = target.to(loc, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        if args.benchmark == 0:
            optimizer.zero_grad()

        if args.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        stream = torch.npu.current_stream()
        stream.synchronize()

        if args.benchmark == 0:
            optimizer.step()
        elif args.benchmark == 1:
            BATCH_SIZE_multiplier = int(OPTIMIZER_BATCH_SIZE / args.batch_size)
            BM_optimizer_step = ((i + 1) % BATCH_SIZE_multiplier) == 0
            if BM_optimizer_step:
                for param_group in optimizer.param_groups:
                    for param in param_group['params']:
                        param.grad /= BATCH_SIZE_multiplier
                optimizer.step()
                optimizer.zero_grad()
        stream = torch.npu.current_stream()
        stream.synchronize()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                        and args.rank % ngpus_per_node == 0):
                progress.display(i)


    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                and args.rank % ngpus_per_node == 0):
        if batch_time.avg > 0:
            print("[npu id:", args.gpu, "]", '* FPS@all {:.3f}, TIME@all {:.3f}'.format(ngpus_per_node * args.batch_size / batch_time.avg, batch_time.avg))


def validate(val_loader, model, criterion, args, ngpus_per_node):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e', start_count_index=0)
    top1 = AverageMeter('Acc@1', ':6.2f', start_count_index=0)
    top5 = AverageMeter('Acc@5', ':6.2f', start_count_index=0)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        loc = 'npu:{}'.format(args.gpu)
        mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).view(1, 3, 1, 1)
        std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).view(1, 3, 1, 1)
        mean = mean.to(loc, non_blocking=True)
        std = std.to(loc, non_blocking=True)

        end = time.time()
        for i, (images, target) in enumerate(val_loader):

            images = images.to(loc, non_blocking=True).to(torch.float).sub(mean).div(std)
            target = target.to(loc, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
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

        # TODO: this should also be done with the ProgressMeter
        if not args.multiprocessing_distributed or \
                (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            print("[npu id:", args.gpu, "]", '[AVG-ACC] * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))
    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best_acc%.4f_epoch%d.pth.tar' % (state['best_acc1'], state['epoch']))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', start_count_index=10):
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
        print("[npu id:", os.environ['LOCAL_DEVICE_ID'], "]", '\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = args.lr * (0.1 ** (epoch // (args.epochs//3 - 3)))

    if args.warm_up_epochs > 0 and epoch < args.warm_up_epochs:
        lr = args.lr * ((epoch + 1) / (args.warm_up_epochs + 1))
    else:
        alpha = 0
        cosine_decay = 0.5 * (
                1 + np.cos(np.pi * (epoch - args.warm_up_epochs) / (args.epochs - args.warm_up_epochs)))
        decayed = (1 - alpha) * cosine_decay + alpha
        lr = args.lr * decayed

    print("=> Epoch[%d] Setting lr: %.4f" % (epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    target = target.to(torch.int32)
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def fast_collate(batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if nump_array.ndim < 3:
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)

        tensor[i] += torch.from_numpy(nump_array)

    return tensor, targets


def get_pytorch_train_loader(data_path, batch_size, workers=5, _worker_init_fn=None, distributed=False):
    traindir = os.path.join(data_path, 'train')
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomHorizontalFlip(),
        ]))

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    dataloader_fn = MultiEpochsDataLoader  # torch.utils.data.DataLoader
    train_loader = dataloader_fn(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=workers, worker_init_fn=_worker_init_fn, pin_memory=True, sampler=train_sampler,
        collate_fn=fast_collate, drop_last=True)
    return train_loader, len(train_loader), train_sampler


def get_pytorch_val_loader(data_path, batch_size, workers=5, _worker_init_fn=None, distributed=False):
    valdir = os.path.join(data_path, 'val')
    val_dataset = datasets.ImageFolder(
        valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]))

    if distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        val_sampler = None

        dataloader_fn = MultiEpochsDataLoader  # torch.utils.data.DataLoader
        val_loader = dataloader_fn(
            val_dataset,
            sampler=val_sampler,
            batch_size=batch_size, shuffle=(val_sampler is None),
            num_workers=workers, worker_init_fn=_worker_init_fn, pin_memory=True, collate_fn=fast_collate)

    return val_loader

if __name__ == '__main__':
    main()
