# -*- coding: utf-8 -*-
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import warnings

warnings.filterwarnings('ignore')
import argparse
import os
import random
import shutil
import time
import torch
import numpy as np
import apex
from apex import amp
import torch.nn as nn
import torch.nn.parallel
import torch.npu
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from ctpn.ctpn import CTPN_Model, RPN_CLS_Loss, RPN_REGR_Loss
from ctpn.dataset import VOCDataset
from ctpn.dataset import ICDARDataset
from ctpn.dataset import icdarDataset
from ctpn import config

# torch.multiprocessing.set_sharing_strategy('file_system')
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
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
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--data-path', default='./dataset/icdar13', type=str,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:50001', type=str,
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
## for ascend 910
parser.add_argument('--device', default='npu', type=str, help='npu or gpu')
parser.add_argument('--addr', default='10.136.181.115',
                    type=str, help='master addr')
parser.add_argument('--device_list', default='0,1,2,3,4,5,6,7',
                    type=str, help='device id list')
parser.add_argument('--amp', default=False, action='store_true',
                    help='use amp to train the model')
parser.add_argument('--loss-scale', default=64., type=float,
                    help='loss scale using in amp, default -1 means dynamic')
parser.add_argument('--opt-level', default='O2', type=str,
                    help='loss scale using in amp, default -1 means dynamic')
parser.add_argument('--prof', default=False, action='store_true',
                    help='use profiling to evaluate the performance of model')
parser.add_argument('--warm_up_epochs', default=5, type=int,
                    help='warm up')


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
    print(args.device_list)

    os.environ['MASTER_ADDR'] = args.addr
    os.environ['MASTER_PORT'] = '29688'
    make_dir('./output_models/')

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.process_device_map = device_id_to_process_device_map(args.device_list)

    if args.device == 'npu':
        ngpus_per_node = len(args.process_device_map)
    else:
        if args.distributed:
            ngpus_per_node = torch.cuda.device_count()
        else:
            ngpus_per_node = 1
    print('ngpus_per_node:', ngpus_per_node)
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = args.process_device_map[gpu]

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

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

    print("=> creating model")
    model = CTPN_Model()
    critetion_cls = RPN_CLS_Loss('cpu')
    critetion_regr = RPN_REGR_Loss('cpu')
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single deviceF scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            if args.device == 'npu':
                loc = 'npu:{}'.format(args.gpu)
                torch.npu.set_device(loc)
                model = model.to(loc)
            else:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)

            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / args.world_size)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        else:
            if args.device == 'npu':
                loc = 'npu:{}'.format(args.gpu)
                model = model.to(loc)
            else:
                model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
    elif args.gpu is not None:
        if args.device == 'npu':
            loc = 'npu:{}'.format(args.gpu)
            torch.npu.set_device(args.gpu)
            model = model.to(loc)
        else:
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)

    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.device == 'npu':
            loc = 'npu:{}'.format(args.gpu)
        else:
            print("before : model = torch.nn.DataParallel(model).cuda()")

    # define loss function (criterion) and optimizer
    optimizer = apex.optimizers.NpuFusedAdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)
    if args.amp:
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.opt_level, loss_scale=args.loss_scale)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            if args.pretrained:
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], broadcast_buffers=False,
                                                                  find_unused_parameters=True)
            else:
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], broadcast_buffers=False)
        else:
            model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.device == 'npu':
            if args.gpu is not None:
                loc = 'npu:{}'.format(args.gpu)
                model = torch.nn.DataParallel(model).to(loc)
        else:
            model = torch.nn.DataParallel(model).cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                if args.device == 'npu':
                    loc = 'npu:{}'.format(args.gpu)
                else:
                    loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if args.amp:
                amp.load_state_dict(checkpoint['amp'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    img_dir = os.path.join(args.data_path, 'Challenge2_Training_Task12_Images/')
    label_dir = os.path.join(args.data_path, 'Challenge2_Training_Task1_GT/')
    # Data loading code
    train_dataset = icdarDataset(config.img_dir, config.label_dir)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(
                train_sampler is None),
        num_workers=args.workers, pin_memory=False, sampler=train_sampler, drop_last=True)

    if args.prof:
        profiling(train_loader, model, critetion_regr, critetion_cls, optimizer, args)
        return

    start_time = time.time()
    all_fps = []
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        cur_fps = train(train_loader, model, critetion_regr, critetion_cls, optimizer, epoch, args, ngpus_per_node)
        if cur_fps is not None:
            all_fps.append(cur_fps)

        if args.device == 'npu' and args.gpu == 0 and epoch == 199:
            print("Complete 200 epoch training, take time:{}h".format(round((time.time() - start_time) / 3600.0, 2)))

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):

            ############## npu modify begin #############
            if args.amp:
                if (epoch + 1) % 5 == 0:
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': 'ctpn',
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'amp': amp.state_dict(),
                    }, filename=f'output_models/checkpoint-{epoch + 1}.pth.tar')
            else:
                if (epoch + 1) % 5 == 0:
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': 'ctpn',
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, filename=f'output_models/checkpoint-{epoch + 1}.pth.tar')
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                and args.rank % ngpus_per_node == 0):
        if all_fps:
            print('overallFPS after training:', np.mean(all_fps))
        ############## npu modify end #############


def profiling(data_loader, model, critetion_regr, critetion_cls, optimizer, args):
    # switch to train mode
    model.train()

    def update(model, images, clss, regrs, optimizer):
        out_cls, out_regr = model(images)
        loss_regr = critetion_regr(out_regr, regrs)
        loss_cls = critetion_cls(out_cls, clss)
        loss = loss_cls.to(loc, non_blocking=True) + loss_regr.to(loc, non_blocking=True)
        if args.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.zero_grad()
        optimizer.step()

    for step, (images, clss, regrs) in enumerate(data_loader):
        if args.device == 'npu':
            loc = 'npu:{}'.format(args.gpu)
            images = images.to(loc, non_blocking=True)
            clss = clss.to(loc, non_blocking=True)
            regrs = regrs.to(loc, non_blocking=True)
        else:
            images = images.cuda(args.gpu, non_blocking=True)
            clss = clss.cuda(args.gpu, non_blocking=True)
            regrs = regrs.cuda(args.gpu, non_blocking=True)

        if step < 5:
            update(model, images, clss, regrs, optimizer)
        else:
            if args.device == 'npu':
                with torch.autograd.profiler.profile(use_npu=True) as prof:
                    update(model, images, clss, regrs, optimizer)
            else:
                with torch.autograd.profiler.profile(use_cuda=True) as prof:
                    update(model, images, clss, regrs, optimizer)
            break
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank == 0):
        prof.export_chrome_trace("output.prof")


def train(train_loader, model, critetion_regr, critetion_cls, optimizer, epoch, args, ngpus_per_node):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses_cls = AverageMeter('LossCls', ':.4e')
    losses_regr = AverageMeter('LossRegr', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses_cls, losses_regr],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, clss, regrs) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.device == 'npu':
            # torch.npu.global_step_inc()
            loc = 'npu:{}'.format(args.gpu)
            images = images.to(loc, non_blocking=True)
            clss = clss.to(loc, non_blocking=True)
            regrs = regrs.to(loc, non_blocking=True)

        # compute output
        out_cls, out_regr = model(images)
        loss_regr = critetion_regr(out_regr, regrs)
        loss_cls = critetion_cls(out_cls, clss)
        loss = loss_cls.to(loc, non_blocking=True) + loss_regr.to(loc, non_blocking=True)

        # measure accuracy and record loss
        losses_regr.update(loss_regr.item(), images.size(0))
        losses_cls.update(loss_cls.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if args.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()
        if args.device == 'npu':
            torch.npu.synchronize()

        # measure elapsed time
        cost_time = time.time() - end
        batch_time.update(cost_time)
        end = time.time()

        if i % args.print_freq == 0:
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                        and args.rank % ngpus_per_node == 0):
                progress.display(i)

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                        and args.rank % ngpus_per_node == 0):
                print("[npu id:", args.gpu, "]", "batch_size:", args.world_size * args.batch_size,
                      'Time: {:.3f}'.format(batch_time.avg), '* FPS@all {:.3f}'.format(
                        args.batch_size * args.world_size / batch_time.avg))
                if i >= 10:
                    cur_fps = args.batch_size * args.world_size / batch_time.avg
                    return cur_fps
                else:
                    return None
            else:
                return None


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', start_count_index=2):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.start_count_index = start_count_index

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if self.count == 0:
            pass

        self.count += n
        if self.count > (self.start_count_index * self.N):
            self.sum += val * n

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

    print("=> Epoch[%d] Setting lr: %.8f" % (epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def make_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)


if __name__ == '__main__':
    main()
