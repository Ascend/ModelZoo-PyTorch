# Copyright 2022 Huawei Technologies Co., Ltd
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
import sys
import random
import shutil
import time
import traceback
import warnings
from collections import OrderedDict

import torch
if torch.__version__ >= '1.8':
    import torch_npu

import torchvision
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
import numpy as np
import moxing

from apex import amp
from apex.optimizers import NpuFusedSGD
from alexnet import AlexNet

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='alexnet',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: alexnet)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=2, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
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
parser.add_argument('--dist-url', default='', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='hccl', type=str,
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
parser.add_argument('--amp', default=True, action='store_true',
                    help='use amp to train the model')
parser.add_argument('--loss-scale', default='dynamic', 
                    help='loss scale using in amp')
parser.add_argument('--opt-level', default='O2', type=str,
                    help='loss scale using in amp, default -1 means dynamic')
parser.add_argument('--prof', default=False, action='store_true',
                    help='use profiling to evaluate the performance of model')

parser.add_argument('--label-smoothing',
                    default=0.1,
                    type=float,
                    metavar='S',
                    help='label smoothing')
parser.add_argument('--warm_up_epochs', default=0, type=int,
                    help='warm up')
# modelarts
parser.add_argument('--data_url', metavar='DIR', default='/cache/data_url',
                    help='path to dataset')
parser.add_argument('--pretrained_pth_url', default='', type=str, metavar='PATH',
                    help='path to pretrained url')
parser.add_argument('--train_url', default='/cache/training', type=str, metavar='PATH',
                    help='setting dir of training output')
parser.add_argument('--onnx', default=True, help='convert pth model to onnx')

best_acc1 = 0
CACHE_TRAINING_URL = "/cache/training"
CACHE_MODEL_URL = '/cache/model_path'
CACHE_DATA_URL = '/cache/data_url'
warnings.filterwarnings('ignore')


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
    print(f"args={args}")

    os.environ['MASTER_ADDR'] = args.addr
    os.environ['MASTER_PORT'] = '41111'

    if args.gpu is None:
        args.gpu = 0

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

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))
        # main_worker(0, ngpus_per_node, args)
    else:
        # Simply call main_worker function
        main_worker(0, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
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
            print(f"args.dist_backend={args.dist_backend}, args.world_size={args.world_size}, args.rank={args.rank}")
            dist.init_process_group(backend=args.dist_backend,  # init_method=args.dist_url,
                                    world_size=args.world_size, rank=args.rank)
        else:
            dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                    world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = torchvision.models.alexnet(pretrained=False)
        print("load your model")

        os.makedirs(CACHE_MODEL_URL, exist_ok=True)
        model_url = os.path.join(CACHE_MODEL_URL, 'checkpoint.pth.tar')
        moxing.file.copy_parallel(args.pretrained_pth_url, model_url)
        pretrained_dict = torch.load(model_url, map_location="cpu")["state_dict"]
        if 'fc.weight' in pretrained_dict:
            try:
                pretrained_dict.pop('module.fc.weight')
                pretrained_dict.pop('module.fc.bias')
            except:
                pretrained_dict.pop('fc.weight')
                pretrained_dict.pop('fc.bias')
        model.load_state_dict(pretrained_dict, strict=False)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = AlexNet()
        print("===============AlexNet()===============: Dropout training on cpu")

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
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
            print("[gpu id:", args.gpu, "]",
                  "============================test   args.gpu is not None   else==========================")
    elif args.gpu is not None:
        print("[gpu id:", args.gpu, "]",
              "============================test   elif args.gpu is not None:==========================")
        if args.device == 'npu':
            loc = 'npu:{}'.format(args.gpu)
            torch.npu.set_device(args.gpu)
            model = model.to(loc)
        else:
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)

    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        print("[gpu id:", args.gpu, "]", "============================test   1==========================")
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            print("[gpu id:", args.gpu, "]", "============================test   2==========================")
        else:
            print("[gpu id:", args.gpu, "]", "============================test   3==========================")
            if args.device == 'npu':
                loc = 'npu:{}'.format(args.gpu)
            else:
                print("before : model = torch.nn.DataParallel(model).cuda()")

    # define loss function (criterion) and optimizer
    optimizer = NpuFusedSGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

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
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            else:
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            print("[gpu id:", args.gpu, "]",
                  "============================test   args.gpu is not None   else==========================")
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        print("[gpu id:", args.gpu, "]",
              "============================test   elif args.gpu is not None:==========================")
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            if args.device == 'npu':
                loc = 'npu:{}'.format(args.gpu)
                model = torch.nn.DataParallel(model).to(loc)
            else:
                model = torch.nn.DataParallel(model).cuda()
                
    if args.device == 'npu':
        loc = 'npu:{}'.format(args.gpu)
        loss = nn.CrossEntropyLoss().to(loc)
        if args.label_smoothing > 0.0:        
            loss = lambda: LabelSmoothing(loc, args.label_smoothing)
        criterion = loss().cuda(args.gpu)
    else:
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)

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
            best_acc1 = checkpoint['best_acc1']
            checkpoint['state_dict'] = proc_node_module(checkpoint, 'state_dict')
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            # print(f"checkpoint['state_dict']={checkpoint['state_dict']}")
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if args.amp:
                amp.load_state_dict(checkpoint['amp'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    if not os.path.exists(CACHE_DATA_URL):
        os.makedirs(CACHE_DATA_URL)
    moxing.file.copy_parallel(args.data_url, CACHE_DATA_URL)
    print(f"training data copy to {CACHE_DATA_URL} finish!")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    # Data loading code
    train_loader, train_loader_len, train_sampler = get_pytorch_train_loader(CACHE_DATA_URL, args.batch_size, normalize,
                                                                       workers=args.workers, distributed=args.distributed)
    val_loader = get_pytorch_val_loader(CACHE_DATA_URL, args.batch_size, normalize, args.workers, distributed=False)

    if args.evaluate:
        validate(val_loader, model, criterion, args, ngpus_per_node)
        return
    
    if args.prof:
        profiling(train_loader, model, criterion, optimizer, args)
        return
    
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, ngpus_per_node)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args, ngpus_per_node)

        # remember best acc@1 and save checkpoint
        is_best = acc1 >= best_acc1
        best_acc1 = max(acc1, best_acc1)
        if args.device == 'npu' and args.gpu == 0 and epoch == 89:
            print("Complete 90 epoch training, take time:{}h".format(round((time.time() - start_time) / 3600.0, 2)))

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            ############## npu modify begin #############
            print(f'args.amp={args.amp}')
            if args.amp:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                    'amp': amp.state_dict(),
                }, is_best, epoch + 1)
            else:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                }, is_best, epoch + 1)
        ############## npu modify end #############
    
    moxing.file.copy_parallel(CACHE_TRAINING_URL, args.train_url)


def profiling(data_loader, model, criterion, optimizer, args):
    # switch to train mode
    model.train()

    def update(model, images, target, optimizer):
        output = model(images)
        loss = criterion(output, target)
        optimizer.zero_grad()
        if args.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

    for step, (images, target) in enumerate(data_loader):
        if args.device == 'npu':
            loc = 'npu:{}'.format(args.gpu)
            images = images.to(loc, non_blocking=True).to(torch.float)
            target = target.to(torch.int32).to(loc, non_blocking=True)
        else:
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            
        if step < 5:
            update(model, images, target, optimizer)
        else:
            if args.device == 'npu':
                with torch.autograd.profiler.profile(use_npu=True) as prof:
                    update(model, images, target, optimizer)
            else:
                with torch.autograd.profiler.profile(use_cuda=True) as prof:
                    update(model, images, target, optimizer)
            break

    prof.export_chrome_trace("output.prof")

class data_prefetcher():
    def __init__(self, loader, args):
        self.args = args
        self.loader = iter(loader)
        if args.device == 'npu':
            self.stream = torch.npu.Stream()
        else:
            self.stream = torch.cuda.Stream()
        
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).view(1,3,1,1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).view(1,3,1,1)

        if args.device == 'npu' and args.gpu is not None:
            self.loc = 'npu:{}'.format(args.gpu)
        elif args.gpu is not None:
            self.loc = 'cuda:{}'.format(args.gpu)

        self.mean = self.mean.to(self.loc, non_blocking=True)
        self.std = self.std.to(self.loc, non_blocking=True)

        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        if self.args.device == 'npu':
            ctx = torch.npu.stream(self.stream)
        else:
            ctx = torch.cuda.stream(self.stream)

        with ctx:
            self.next_input = self.next_input.to(self.loc, non_blocking=True)
            self.next_target = self.next_target.to(self.loc, non_blocking=True)
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    
    def next(self):
        if self.args.device == 'npu':
            torch.npu.current_stream().wait_stream(self.stream)
        else:
            torch.cuda.current_stream().wait_stream(self.stream)

        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target

def train(train_loader, model, criterion, optimizer, epoch, args, ngpus_per_node):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    if args.device == 'npu' and args.gpu is not None:
        loc = 'npu:{}'.format(args.gpu)
    elif args.gpu is not None:
        loc = 'cuda:{}'.format(args.gpu)

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.to(loc, non_blocking=True)
        target = target.to(torch.int32).to(loc, non_blocking=True)

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
        if args.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

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
        if batch_time.avg == 0:
            print("[npu id:", args.gpu, "]", "batch_size:", args.batch_size * ngpus_per_node,
                'Time: {:.3f}'.format(batch_time.avg), '* FPS@all {:.3f}'.format(
                    ngpus_per_node * args.batch_size / (batch_time.avg+0.0001)))
        else:
            print("[npu id:", args.gpu, "]", "batch_size:", args.batch_size * ngpus_per_node,
                'Time: {:.3f}'.format(batch_time.avg), '* FPS@all {:.3f}'.format(
                    ngpus_per_node * args.batch_size / batch_time.avg))


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
        if args.device == 'npu' and args.gpu is not None:
            loc = 'npu:{}'.format(args.gpu)
        elif args.gpu is not None:
            loc = 'cuda:{}'.format(args.gpu)

        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(loc, non_blocking=True)
            target = target.to(torch.int32).to(loc, non_blocking=True)

            # compute output
            output = model(images)
            
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            cost_time = time.time() - end
            batch_time.update(cost_time)
            end = time.time()

            if i % args.print_freq == 0:
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                            and args.rank % ngpus_per_node == 0):
                    progress.display(i)

        if i % args.print_freq == 0:
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                        and args.rank % ngpus_per_node == 0):
                print("[gpu id:", args.gpu, "]", '[AVG-ACC] * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                      .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, epoch, filename='checkpoint.pth.tar'):
    path_dir = CACHE_TRAINING_URL
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    c_filename= os.path.join(path_dir, "checkpoint.pth.tar")
    b_filename= os.path.join(path_dir, "model_best.pth.tar")
    torch.save(state, c_filename)
    if is_best:
        print(f"========== save best pth, epoch is {epoch} ================")
        moxing.file.copy(c_filename, b_filename)
        onnx_file = os.path.join(path_dir, "alexnet.onnx")
        convert(b_filename, onnx_file)


def convert(pth_file, onnx_file):
    checkpoint = torch.load(pth_file, map_location='cpu')
    checkpoint['state_dict'] = proc_node_module(checkpoint, 'state_dict')
    model = torchvision.models.alexnet(pretrained=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    input_names = ["actual_input_1"]
    output_names = ["output1"]
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy_input, onnx_file, input_names=input_names, output_names=output_names,
                      opset_version=11)


def proc_node_module(checkpoint, AttrName):
    new_state_dict = OrderedDict()
    for k, v in checkpoint[AttrName].items():
        if(k[0:7] == "module."):
            name = k[7:]
        else:
            name = k[0:]
        new_state_dict[name] = v
    return new_state_dict

        
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', start_count_index=2):
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

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if args.warm_up_epochs > 0 and epoch < args.warm_up_epochs:
        lr = args.lr * ((epoch + 1) / (args.warm_up_epochs + 1))
    else:
        alpha = 0
        cosine_decay = 0.5 * (
                1 + np.cos(np.pi * (epoch - args.warm_up_epochs) / (args.epochs - args.warm_up_epochs)))
        decayed = (1 - alpha) * cosine_decay + alpha
        lr = args.lr * decayed

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, loc, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.

        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.device = loc

    def forward(self, x, target):
        target = target.to(torch.int64)

        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1).to(torch.int64))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

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

def get_pytorch_train_loader(data_path, batch_size, normalize, workers=5, _worker_init_fn=None, distributed=False):
    traindir = os.path.join(data_path, 'train')
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=workers, pin_memory=True, sampler=train_sampler)
    
    return train_loader, len(train_loader), train_sampler

def get_pytorch_val_loader(data_path, batch_size, normalize, workers=5, _worker_init_fn=None, distributed=False):
    valdir = os.path.join(data_path, 'val')
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    )
    
    if distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        val_sampler = None
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        sampler=val_sampler, 
        batch_size=batch_size,
        shuffle=(val_sampler is None),
        num_workers=workers, 
        pin_memory=True)

    return val_loader


if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    import traceback
    try:
        main()
    except Exception as e:
        traceback.print_exc()
