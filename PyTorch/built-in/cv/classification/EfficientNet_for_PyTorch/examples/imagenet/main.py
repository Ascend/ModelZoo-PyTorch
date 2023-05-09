# Copyright [yyyy] [name of copyright owner]
# Copyright 2020 Huawei Technologies Co., Ltd
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

import argparse
import os
import re
import sys
import time
import PIL
import numpy as np

import torch
if torch.__version__ >= "1.8":
    import torch_npu
import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import apex
from apex import amp
try:
    from torch_npu.utils.profiler import Profile
except ImportError:
    print("Profile not in torch_npu.utils.profiler now... Auto Profile disabled.", flush=True)
    class Profile:
        def __init__(self, *args, **kwargs):
            pass
        def start(self):
            pass
        def end(self):
            pass

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),'../../'))
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch import rand_augment_transform, augment_and_mix_transform, auto_augment_transform

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    help='model architecture (default: resnet18)')
parser.add_argument('-j', '--workers', default=64, type=int, metavar='N',
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
parser.add_argument('--wd', '--weight-decay', default=1e-5, type=float,
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
parser.add_argument('--pretrained_weight', default='', type=str, metavar='PATH',
                    help='path to pretrained weight')
parser.add_argument('--num_classes', default=1000, type=int,
                    help='number of class')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='hccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--npu', default=None, type=str,
                    help='npu id to use.')
parser.add_argument('--image_size', default=224, type=int,
                    help='image size')
parser.add_argument('--advprop', default=False, action='store_true',
                    help='use advprop or not')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--autoaug', action='store_true', help='use auto augment')
parser.add_argument('--amp', action='store_true', help='use apex')
parser.add_argument('--pm', '--precision-mode', default='O1', type=str,
                    help='precision mode to use for mix precision, only support O1, O2')
parser.add_argument('--loss_scale', default=1024, type=int, help='loss_scale for amp')
parser.add_argument('--addr', default='127.0.0.1', type=str,
                    help='npu id to use.')
parser.add_argument('--nnpus_per_node', default=None, type=int,
                    help='number of npus to use for distributed train on each node')
parser.add_argument('--val_feq', default=10, type=int,
                    help='validation frequency')
parser.add_argument('--device_list', default='0,1,2,3,4,5,6,7', type=str, help='device id list')
parser.add_argument('--stop-step-num', default=None, type=int,
                    help='after the stop-step, killing the training task')
parser.add_argument('--prof', action='store_true',
                    help='use profiling to evaluate the performance of model')

cur_step = 0

# for servers to immediately record the logs
#def flush_print(func):
    #def new_print(*args, **kwargs):
        #func(*args, **kwargs)
        #sys.stdout.flush()
    #return new_print
#print = flush_print(print)

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

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
	
    args.process_device_map = device_id_to_process_device_map(args.device_list)
    nnpus_per_node = len(args.process_device_map)

    
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = nnpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        os.environ['MASTER_ADDR'] = args.addr
        os.environ['MASTER_PORT'] = '29688'
        mp.spawn(main_worker, nprocs=nnpus_per_node, args=(nnpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.npu, nnpus_per_node, args)

def main_worker(npu, nnpus_per_node, args):
    args.npu = npu
    global cur_step
    if args.distributed:
        args.npu = args.process_device_map[npu]

    if args.npu is not None:
        print("Use npu: {} for training".format(args.npu))
        torch.npu.set_device('npu:' + str(args.npu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * nnpus_per_node + int(npu)

        dist.init_process_group(backend=args.dist_backend,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if 'efficientnet' in args.arch:  # NEW
        if args.pretrained:
            model = EfficientNet.from_pretrained(args.arch, advprop=args.advprop, weights_path=args.pretrained_weight, num_classes=args.num_classes)
            print("=> using pre-trained model '{}'".format(args.arch))
        else:
            print("=> creating model '{}'".format(args.arch))
            model = EfficientNet.from_name(args.arch)

    else:
        if args.pretrained:
            print("=> using pre-trained model '{}'".format(args.arch))
            model = models.__dict__[args.arch](pretrained=True)
        else:
            print("=> creating model '{}'".format(args.arch))
            model = models.__dict__[args.arch]()

    criterion = nn.CrossEntropyLoss().to('npu:' + str(args.npu))

    optimizer = apex.optimizers.NpuFusedSGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    model = model.to('npu:' + str(args.npu))
    if args.amp:
        print("=> use amp...")
        if args.pm not in ['O1', 'O2']:
            print('=>unsupported precision mode!')
            exit()
        opt_level = args.pm
        model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level, loss_scale=args.loss_scale,combine_grad=True)

    global total_batch_size
    total_batch_size = args.batch_size
    if args.distributed:
        args.batch_size = int(args.batch_size / nnpus_per_node)
        args.workers = int(args.workers / nnpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.npu], broadcast_buffers=False)



    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='npu:' + str(args.npu))
            args.start_epoch = checkpoint['epoch']
            if args.amp:
                amp.load_state_dict(checkpoint['amp'])
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    if args.advprop:
        normalize = transforms.Lambda(lambda img: img * 2.0 - 1.0)
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    if 'efficientnet' in args.arch:
        image_size = EfficientNet.get_image_size(args.arch)
    else:
        image_size = args.image_size

    if args.autoaug:
        print("=> use auto augment...")
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                auto_augment_wrapper(image_size),
                transforms.ToTensor(),
                normalize,
            ]))
    else:
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    val_transforms = transforms.Compose([
        transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])
    print('npu:' + str(args.npu), ' optimizer params:', optimizer)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, val_transforms),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        res = validate(val_loader, model, criterion, args, nnpus_per_node)
        with open('res.txt', 'w') as f:
            print(res, file=f)
        return

    if args.prof:
        profiling(train_loader, model, criterion, optimizer, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, nnpus_per_node)

        # evaluate on validation set
        if epoch % args.val_feq == 0 or epoch == args.epochs - 1:
            validate(val_loader, model, criterion, args, nnpus_per_node)

        if epoch == args.epochs - 1:
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % nnpus_per_node == 0):
                if not args.amp:
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    })
                else:
                    save_checkpoint({
                       'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'amp': amp.state_dict(),
                    })
        if args.stop_step_num is not None and cur_step >= args.stop_step_num:
            pass


def profiling(data_loader, model, criterion, optimizer, args):
    # switch to train mode
    model.train()

    def update(model, images, target, optimizer):
        output = model(images)
        loss = criterion(output, target)
        if args.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.zero_grad()
        optimizer.step()

    for step, (images, target) in enumerate(data_loader):

        loc = 'npu:{}'.format(args.npu)
        images = images.to(loc, non_blocking=True).to(torch.float)
        target = target.to(torch.int32).to(loc, non_blocking=True)

        if step < 5:
            update(model, images, target, optimizer)
        else:
            with torch.autograd.profiler.profile(use_npu=True) as prof:
                update(model, images, target, optimizer)
            break

    prof.export_chrome_trace("output.prof")

def train(train_loader, model, criterion, optimizer, epoch, args, nnpus_per_node):
    global cur_step
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':6.4f')
    lr = AverageMeter('LR', ':6.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    fps_time = AverageMeter('FPS', ':6.1f')
    progress = ProgressMeter(len(train_loader), fps_time, batch_time, data_time, losses, lr, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    step_per_epoch = len(train_loader)
    profiler = Profile(start_step=int(os.getenv("PROFILE_START_STEP", 10)),
                       profile_type=os.getenv("PROFILE_TYPE"))
    for i, (images, target) in enumerate(train_loader):
        if i > 100:
            pass
        cur_step = epoch * step_per_epoch + i
        adjust_learning_rate_fraction_epoch(optimizer, epoch, args)

        # measure data loading time
        data_time.update(time.time() - end)

        optimizer.zero_grad()

        target = target.int()
        images, target = images.to('npu:' + str(args.npu), non_blocking=True), target.to('npu:' + str(args.npu), non_blocking=True)

        profiler.start()
        # compute output
        output = model(images)

        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        losses.update(loss.item(), images.size(0))
        lr.update(optimizer.param_groups[0]['lr'], images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        # compute gradient and do SGD step

        if args.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        profiler.end()

        # measure elapsed time
        fps_time.update(total_batch_size / (time.time() - end))
        batch_time.update(time.time() - end)
        end = time.time()
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % nnpus_per_node == 0):
            progress.print(i)
        if args.stop_step_num is not None and cur_step >= args.stop_step_num:
            break

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                        and args.rank % nnpus_per_node == 0):
        fps = str(fps_time)
        p1 = re.compile(r'[(](.*?)[)]', re.S)
        FPS = re.findall(p1, fps)[0]
        print(' * FPS@all {}'.format(FPS))

def validate(val_loader, model, criterion, args, nnpus_per_node):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if i > 10:
                pass
            target = target.int()
            images, target = images.to('npu:' + str(args.npu), non_blocking=True), target.to('npu:' + str(args.npu), non_blocking=True)

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

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                        and args.rank % nnpus_per_node == 0):
                progress.print(i)

        # TODO: this should also be done with the ProgressMeter
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % nnpus_per_node == 0):

            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        

    return top1.avg


def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.skip = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.skip = 0

    def update(self, val, n=1):
        self.val = val
        # the first 5 value are not accumulated in the average stats
        self.skip += 1
        if self.skip < 5:
            return
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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

def auto_augment_wrapper(img_size, auto_augment='original-mstd0.5'):
    IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
    assert isinstance(auto_augment, str)
    aa_params = dict(
        translate_const=int(img_size * 0.45),
        img_mean=tuple([min(255, round(255 * x)) for x in IMAGENET_DEFAULT_MEAN]),
    )
    if auto_augment.startswith('rand'):
        return rand_augment_transform(auto_augment, aa_params)
    elif auto_augment.startswith('augmix'):
        aa_params['translate_pct'] = 0.3
        return augment_and_mix_transform(auto_augment, aa_params)
    else:
        return auto_augment_transform(auto_augment, aa_params)

def adjust_learning_rate_fraction_epoch(optimizer, epoch, args):
    """Use the epoch cosine schedule"""

    alpha = 0
    cosine_decay = 0.5 * (1 + np.cos(np.pi * epoch / args.epochs))
    decayed = (1 - alpha) * cosine_decay + alpha
    lr = args.lr * decayed
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    main()
