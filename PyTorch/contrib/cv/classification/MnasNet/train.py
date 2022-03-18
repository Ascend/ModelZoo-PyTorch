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

import argparse
import os
import random
import shutil
import time
import warnings
import math
import numpy as np

import torch
import apex
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import mnasnet
import torch.npu

CALCULATE_DEVICE = "npu:0"

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--device', default='npu', type=str, help='npu or gpu')
parser.add_argument('--device-list', default='0,1,2,3,4,5,6,7', type=str, help='device id list')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all NPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

parser.add_argument('--label-smoothing', '--ls', default=0.1, type=float)

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
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--npu', default=None, type=int,
                    help='NPU id to use.')
parser.add_argument('--warmup', default=0, type=int,
                    help='Warmup epochs.')
parser.add_argument('--local_rank', default=0, type=int,
                    help="rank id of process")
parser.add_argument('--run-prof', action='store_true', help='only for prof')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N NPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')


best_acc1 = 0


def main():
    args = parser.parse_args()


    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.npu is not None:
        warnings.warn('You have chosen a specific NPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    # ngpus_per_node = torch.cuda.device_count()
    args.process_device_map = device_id_to_process_device_map(args.device_list)
    if args.device == 'npu':
        ngpus_per_node = len(args.process_device_map)
    else:
        ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        main_worker(args.local_rank, ngpus_per_node, args)
    else:
        # Simply call main_worker function
        main_worker(args.npu, ngpus_per_node, args)


def main_worker(npu, ngpus_per_node, args):
    global best_acc1
    # args.npu = npu

    args.npu = args.process_device_map[npu]

    if args.npu is not None:
        print("Use NPU: {} for training".format(args.npu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + npu
            if args.device == 'npu':
                os.environ['MASTER_ADDR'] = '127.0.0.1'  # args.addr
                os.environ['MASTER_PORT'] = '29688'
                dist.init_process_group(backend=args.dist_backend,  # init_method=args.dist_url,
                                        world_size=args.world_size, rank=args.rank)
            else:
                dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                        world_size=args.world_size, rank=args.rank)
            # dist.init_process_group(backend=args.dist_backend, world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        # model = models.__dict__[args.arch](pretrained=True)
        pretrained_dict = torch.load("./model_best.pth.tar", map_location="cpu")["state_dict"]
        model = mnasnet.mnasnet1_0()
        if "fc.weight" in pretrained_dict:
            pretrained_dict.pop("fc.weight")
            pretrained_dict.pop("fc.bias")
        if "module.fc.weight" in pretrained_dict:
            pretrained_dict.pop("module.fc.weight")
            pretrained_dict.pop("module.fc.bias")
        model.load_state_dict(pretrained_dict, strict=False)
    else:
        print("=> creating model '{}'".format('mansnet'))
        model = mnasnet.mnasnet1_0()
        # model = models.__dict__[args.arch]()

    args.loss_scale = 128

    loc = 'npu:{}'.format(args.npu)
    torch.npu.set_device(loc)
    # 计算用于训练的batch_size和workers
    args.batch_size = int(args.batch_size / ngpus_per_node)
    # args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), #Too slow
            normalize,
        ]))
    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))

    train_sampler = None
    val_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    #     val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=False,
        sampler=train_sampler,
        # collate_fn=fast_collate,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False,
        sampler=val_sampler,
        # collate_fn=fast_collate,
        drop_last=True)

    model = model.to(loc)
    # define loss function (criterion) and optimizer
    # criterion = nn.CrossEntropyLoss().to(loc)
    criterion = LabelSmoothingCrossEntropy().to(loc)

    optimizer = apex.optimizers.NpuFusedSGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    nesterov=True)
    lr_schedule = CosineWithWarmup(optimizer, args.warmup, 0.1, args.epochs)

    model, optimizer = amp.initialize(model, optimizer, opt_level="O1", loss_scale=args.loss_scale)
    if args.multiprocessing_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.npu], broadcast_buffers=False)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            amp.load_state_dict(checkpoint['amp'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        lr_schedule.step(epoch)
        if args.rank == 0:
            print('lr = ', lr_schedule.get_lr()[0])
            file = open('log.txt', 'a')
            print('lr = ', lr_schedule.get_lr()[0], file=file)
            file.close()

        if args.run_prof:
            runprof(train_loader, model, criterion, optimizer, epoch, args)
            print('output to output.prof')
            return

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                'amp': amp.state_dict(),
            }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    FPS = AverageMeter('FPS', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5, FPS],
        prefix="Epoch: [{}]".format(epoch),
        fpath='./log.txt')

    # switch to train mode
    model.train()

    end = time.time()
    # prefetcher = data_prefetcher(train_loader)
    # images, target = prefetcher.next()
    # i = -1
    # while images is not None:
    for i, (images, target) in enumerate(train_loader):
        # i += 1
        # measure data loading time
        data_time.update(time.time() - end)

        loc = 'npu:{}'.format(args.npu)
        target = target.to(torch.int32)
        images, target = images.to(loc, non_blocking=False), target.to(loc, non_blocking=False)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        #loss.backward()

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        optimizer.step()
        # measure elapsed time
        batch_time_nw = time.time() - end;
        if i >= 5:
            batch_time.update(batch_time_nw)
        if i >= 2:
            batch_size = images.size(0)
            FPS.update(batch_size / batch_time_nw * args.world_size)

        end = time.time()
        if i % args.print_freq == 0 and args.rank == 0:
            progress.display(i)
        # images, target = prefetcher.next()
    print('NPU: {}, solve {} batchs'.format(args.rank, i))


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ',
        fpath='./log.txt')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()

        # prefetcher = data_prefetcher(val_loader)
        # images, target = prefetcher.next()
        # i = -1
        # while images is not None:
        for i, (images, target) in enumerate(val_loader):
        #     i += 1
            loc = 'npu:{}'.format(args.npu)
            target = target.to(torch.int32)
            images, target = images.to(loc, non_blocking=False), target.to(loc, non_blocking=False)

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

            if i % args.print_freq == 0 and args.rank == 0:
                progress.display(i)
            # images, target = prefetcher.next()

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        file = open('log.txt', 'a')
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5),
              file=file)
        file.close()
    return top1.avg


def runprof(train_loader, model, criterion, optimizer, epoch, args):
    # switch to train mode
    model.train()
    prefetcher = data_prefetcher(train_loader)
    images, target = prefetcher.next()
    i = -1
    while images is not None:
        i += 1
        if args.npu is not None:
            images = images.cuda(args.npu, non_blocking=True)

        if 'npu' in CALCULATE_DEVICE:
            target = target.to(torch.int32)
        images, target = images.to(CALCULATE_DEVICE, non_blocking=True), target.to(CALCULATE_DEVICE, non_blocking=True)

        if i >= 5:
            with torch.autograd.profiler.profile(use_cuda=True) as prof:
                out = model(images)
                loss = criterion(out, target)
                optimizer.zero_grad()
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                optimizer.step()
            prof.export_chrome_trace("output.prof")  # "output.prof"为输出文件地址
            return
        else:
            output = model(images)
            loss = criterion(output, target)
            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
        images, target = prefetcher.next()


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()
        return loss * self.eps / c + (1 - self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", fpath=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        if fpath is not None:
            self.file = open(fpath, 'a')

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        if self.file is not None:
            self.file.write('\t'.join(entries))
            self.file.write('\n')
            self.file.flush()

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

    def close(self):
        if self.file is not None:
            self.file.close()


class CosineWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    """ Implements a schedule where the first few epochs are linear warmup, and
    then there's cosine annealing after that."""

    def __init__(self, optimizer: torch.optim.Optimizer, warmup_len: int,
                 warmup_start_multiplier: float, max_epochs: int,
                 last_epoch: int = -1):
        if warmup_len < 0:
            raise ValueError("Warmup can't be less than 0.")
        self.warmup_len = warmup_len
        if not (0.0 <= warmup_start_multiplier <= 1.0):
            raise ValueError(
                "Warmup start multiplier must be within [0.0, 1.0].")
        self.warmup_start_multiplier = warmup_start_multiplier
        if max_epochs < 1 or max_epochs < warmup_len:
            raise ValueError("Max epochs must be longer than warm-up.")
        self.max_epochs = max_epochs
        self.cosine_len = self.max_epochs - self.warmup_len
        self.eta_min = 0.0  # Final LR multiplier of cosine annealing
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch > self.max_epochs:
            raise ValueError(
                "Epoch may not be greater than max_epochs={}.".format(
                    self.max_epochs))
        if self.last_epoch < self.warmup_len or self.cosine_len == 0:
            # We're in warm-up, increase LR linearly. End multiplier is implicit 1.0.
            slope = (1.0 - self.warmup_start_multiplier) / self.warmup_len
            lr_multiplier = self.warmup_start_multiplier + slope * self.last_epoch
        else:
            # We're in the cosine annealing part. Note that the implementation
            # is different from the paper in that there's no additive part and
            # the "low" LR is not limited by eta_min. Instead, eta_min is
            # treated as a multiplier as well. The paper implementation is
            # designed for SGDR.
            cosine_epoch = self.last_epoch - self.warmup_len
            lr_multiplier = self.eta_min + (1.0 - self.eta_min) * (
                    1 + math.cos(math.pi * cosine_epoch / self.cosine_len)) / 2
        assert lr_multiplier >= 0.0
        return [base_lr * lr_multiplier for base_lr in self.base_lrs]


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
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def device_id_to_process_device_map(device_list):
    devices = device_list.split(",")
    devices = [int(x) for x in devices]
    devices.sort()

    process_device_map = dict()
    for process_id, device_id in enumerate(devices):
        process_device_map[process_id] = device_id

    return process_device_map


def fast_collate(batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        tens = torch.from_numpy(nump_array)
        if (nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)

        tensor[i] += torch.from_numpy(nump_array)

    return tensor, targets


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.npu.Stream()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).npu().view(1, 3, 1, 1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).npu().view(1, 3, 1, 1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.npu.stream(self.stream):
            self.next_input = self.next_input.npu(non_blocking=True)
            self.next_target = self.next_target.npu(non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.npu.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target

if __name__ == '__main__':
    if 'npu' in CALCULATE_DEVICE:
        torch.npu.set_device(CALCULATE_DEVICE)
    main()
