# -*- coding: utf-8 -*-
# Copyright (c) Soumith Chintala 2016,
# All rights reserved.
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

import numpy as np

import torch
if torch.__version__ >= "1.8":
    import torch_npu
import torch.nn as nn
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

from mobilenet import mobilenet_v2
import apex
from apex import amp

from multi_epochs_dataloader import MultiEpochsDataLoader,NoProfiling
try:
    from torch_npu.utils.profiler import Profile
except Exception:
    print("Profile not in torch_npu.utils.profiler now.. Auto Profile disabled.", flush=True)
    class Profile:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def end(self):
            pass

BATCH_SIZE = 6144
OPTIMIZER_BATCH_SIZE = 6144


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default='/opt/npu/dataset/imagenet',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=BATCH_SIZE, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--workspace', type=str, default='./', metavar='DIR',
                    help='path to directory where checkpoints will be stored')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-ef', '--eval-freq', default=5, type=int,
                    metavar='N', help='evaluate frequency (default: 5)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrain', default='', type=str, metavar='PATH',
                    help='path to pretrain model')
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
                    metavar='N', help='set benchmark status (default: 1,run benchmark)')
parser.add_argument('--device', default='npu', type=str, help='npu or gpu')
parser.add_argument('--addr', default='10.136.181.115', type=str, help='master addr')
parser.add_argument('--checkpoint-nameprefix', default='checkpoint', type=str, help='checkpoint-nameprefix')
parser.add_argument('--checkpoint-freq', default=0, type=int,
                    metavar='N', help='checkpoint frequency (default: 0)'
                                      '0: save only one file whitch per epoch;'
                                      'n: save diff file per n epoch'
                                      '-1:no checkpoint,not support')

parser.add_argument('--device-list', default='0,1,2,3,4,5,6,7', type=str, help='device id list')
parser.add_argument('--bin_mode', default=0, type=int, help='Use rt2 in the model: 0->not bin,1->bin')
parser.add_argument('--start_step', default=0, type=int, help='start_step')
parser.add_argument('--stop_step', default=20, type=int, help='stop_step')
parser.add_argument('--profiling', type=str, default='None',help='choose profiling way--CANN,GE,NONE')
parser.add_argument('--train_performance', type=int, default=0, help='train performace')
parser.add_argument('--steps_per_epoch', type=int, default=1000,help='steps per epoch')
parser.add_argument('--port', type=str, default='59629', help='port')

# apex
parser.add_argument('--amp', default=False, action='store_true',
                    help='use amp to train the model')
parser.add_argument('--loss-scale', default=64., type=float,
                    help='loss scale using in amp, default -1 means dynamic')
parser.add_argument('--opt-level', default='O2', type=str,
                    help='loss scale using in amp, default -1 means dynamic')
parser.add_argument('--class-nums', default=1000, type=int, help='class-nums only for pretrain')

# 图模式
parser.add_argument('--graph_mode',
                    action='store_true',
                    help='whether to enable graph mode.')

warnings.filterwarnings('ignore')
best_acc1 = 0

def device_id_to_process_device_map(device_list):
    devices = device_list.split(",")
    devices = [int(x) for x in devices]
    devices.sort()

    process_device_map = dict()
    for process_id, device_id in enumerate(devices):
        process_device_map[process_id] = device_id

    return process_device_map


def main():
    args = parser.parse_args()
    print("===============main()=================")
    print(args)
    print("===============main()=================")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    os.environ['MASTER_ADDR'] = args.addr
    os.environ['MASTER_PORT'] = args.port

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    args.process_device_map = device_id_to_process_device_map(args.device_list)

    if args.device == 'npu':
        # ngpus_per_node = torch.npu.device_count()
        ngpus_per_node = len(args.process_device_map)
    else:
        ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        # The child process uses the environment variables of the parent process,
        # we have to set KERNEL_NAME_ID for every proc
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
    args.gpu = args.process_device_map[gpu]
    if args.bin_mode:
        torch.npu.set_compile_mode(jit_compile=False)
        print("use rt2+bin train model")

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

    args.batch_size = int(args.batch_size / ngpus_per_node)
    args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)

    print("[npu id:", args.gpu, "]", "===============main_worker()=================")
    print("[npu id:", args.gpu, "]", args)
    print("[npu id:", args.gpu, "]", "===============main_worker()=================")

    # Data loading code
    train_loader, train_loader_len, train_sampler = get_pytorch_train_loader(args.data,
                                                                             args.batch_size,
                                                                             workers=args.workers,
                                                                             distributed=args.distributed)

    val_loader = get_pytorch_val_loader(args.data, args.batch_size, args.workers, distributed=False)

    # create model
    print("[npu id:", args.gpu, "]", "=> creating model '{}'".format('mobilenetv2'))
    model = mobilenet_v2(num_classes=args.class_nums)
    model = model.to(loc)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(loc)
    optimizer = apex.optimizers.NpuFusedSGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level, loss_scale=args.loss_scale,combine_grad=True)

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
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.pretrain:
        if not os.path.isfile(args.pretrain):
            print("no chechpoint found at {}".format(args.pretrain))

        print("loading checkpoint '{}'".format(args.pretrain))
        pretrained_dict = torch.load(args.pretrain, map_location="cpu")['state_dict']
        pretrained_dict.pop('module.classifier.1.weight')
        pretrained_dict.pop('module.classifier.1.bias')
        model.load_state_dict(pretrained_dict, strict=False)
        print("loaded checkpoint '{}'".format(args.pretrain))

    cudnn.benchmark = True

    if args.evaluate:
        validate(val_loader, model, criterion, args, ngpus_per_node)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, train_loader_len, model, criterion, optimizer, epoch, args, ngpus_per_node)

        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            # evaluate on validation set
            acc1 = validate(val_loader, model, criterion, args, ngpus_per_node)

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            if not args.multiprocessing_distributed or \
                    (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0 and epoch == args.epochs - 1):
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


def train(train_loader, train_loader_len, model, criterion, optimizer, epoch, args, ngpus_per_node):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
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

    # steps_per_epoch = len(train_loader)
    steps_per_epoch = train_loader_len
    profile = Profile(start_step=int(os.getenv('PROFILE_START_STEP', 10)),
                      profile_type=os.getenv('PROFILE_TYPE'))
    print('==========step per epoch======================', steps_per_epoch)
    for i, (images, target) in enumerate(train_loader):
        #图模式
        if args.graph_mode:
            print("graph mode on")
            torch.npu.enable_graph_mode()
        # measure data loading time
        data_time.update(time.time() - end)

        profile.start()
        global_step = epoch * steps_per_epoch + i
        lr = adjust_learning_rate(optimizer, global_step, steps_per_epoch, args)
        #图模式
        if args.graph_mode:
            images = images.to(loc, non_blocking = True)
            target = target.to(loc, non_blocking = True)
            images = images.to(torch.float).sub(mean).div(std)
            target = target.to(torch.int32)
            # compute output
            output = model(images)
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
        else:
            target = target.to(torch.int32)
            images = images.to(loc, non_blocking=True).to(torch.float).sub(mean).div(std)
            target = target.to(loc, non_blocking=True)
            # compute output
            output = model(images)
            stream = torch.npu.current_stream()
            stream.synchronize()

            loss = criterion(output, target)
            stream = torch.npu.current_stream()
            stream.synchronize()

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
        #图模式
        if not args.graph_mode:
            stream = torch.npu.current_stream()
            stream.synchronize()

        if args.benchmark == 0:
            optimizer.step()
        elif args.benchmark == 1:
            batch_size_multiplier = int(OPTIMIZER_BATCH_SIZE / args.batch_size)
            bm_optimizer_step = ((i + 1) % batch_size_multiplier) == 0
            if bm_optimizer_step:
                for param_group in optimizer.param_groups:
                    for param in param_group['params']:
                        param.grad /= batch_size_multiplier
                optimizer.step()
                optimizer.zero_grad()
        #图模式
        if args.graph_mode:
            torch.npu.launch_graph()
            if i == len(train_loader):
                torch.npu.synchronize()
        else:
            stream = torch.npu.current_stream()
            stream.synchronize()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                        and args.rank % ngpus_per_node == 0):
                progress.display(i)
        profile.end()
    #图模式
    if args.graph_mode:
        print("graph mode off")
        torch.npu.disable_graph_mode()
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

    with torch.no_grad():
        loc = 'npu:{}'.format(args.gpu)
        mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).view(1, 3, 1, 1)
        std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).view(1, 3, 1, 1)
        mean = mean.to(loc, non_blocking=True)
        std = std.to(loc, non_blocking=True)

        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if i > 48 :
                pass

            target = target.to(torch.int32)
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

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.start_count_index = 10

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if self.count == 0:
            self.batchsize = n

        self.val = val
        self.count += n
        if self.count > (self.start_count_index * self.batchsize):
            self.sum += val * n
            self.avg = self.sum / (self.count - self.start_count_index * self.batchsize)

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
        print("[npu id:", '0', "]", '\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, global_step, steps_per_epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = args.lr * (0.98 ** (epoch / 2.5))
    lr = args.lr * (0.98 ** (global_step // int(steps_per_epoch * 2.5)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""

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
    train_dataset = datasets.ImageFolder(traindir,
                                         transforms.Compose([
                                             transforms.RandomResizedCrop(224),
                                             transforms.RandomHorizontalFlip(),
                                         ]))

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    dataloader_fn = MultiEpochsDataLoader
    train_loader = dataloader_fn(train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
                                 num_workers=workers, worker_init_fn=_worker_init_fn, pin_memory=True,
                                 sampler=train_sampler, collate_fn=fast_collate, drop_last=True)
    return train_loader, len(train_loader), train_sampler


def get_pytorch_val_loader(data_path, batch_size, workers=5, _worker_init_fn=None, distributed=False):
    valdir = os.path.join(data_path, 'val')
    val_dataset = datasets.ImageFolder(valdir,
                                       transforms.Compose([
                                           transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                       ]))

    if distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        val_sampler = None

        dataloader_fn = MultiEpochsDataLoader
        val_loader = dataloader_fn(
            val_dataset,
            sampler=val_sampler,
            batch_size=batch_size, shuffle=(val_sampler is None),
            num_workers=workers, worker_init_fn=_worker_init_fn, pin_memory=True, collate_fn=fast_collate, drop_last=True)

    return val_loader


if __name__ == '__main__':
    main()
