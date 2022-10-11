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
import random
import shutil
import time
import warnings
import moxing as mox

import apex
import numpy as np
import torch.npu
from apex import amp

from collections import OrderedDict
import torch
import torch.onnx
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
import senet
CALCULATE_DEVICE = "npu:0"
model_names = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', default='', type=str,
                    help='path to dataset')
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
parser.add_argument('--pretrained', default=False, dest='pretrained', action='store_true',
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
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--npu', default=None, type=int,
                    help='NPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--device', default='npu', type=str, help='npu or gpu')
parser.add_argument('--addr', default='10.136.181.115',
                    type=str, help='master addr')
parser.add_argument('--amp', default=False, action='store_true',
                    help='use amp to train the model')
parser.add_argument('--warm_up_epochs', default=0, type=int,
                    help='warm up')
parser.add_argument('--loss-scale', default=1024., type=float,
                    help='loss scale using in amp, default -1 means dynamic')
parser.add_argument('--opt-level', default='O2', type=str,
                    help='loss scale using in amp, default -1 means dynamic')
parser.add_argument('--prof', default=False, action='store_true',
                    help='use profiling to evaluate the performance of model')
parser.add_argument('--save_path', default='', type=str,
                    help='path to save models')
parser.add_argument('--num_classes', default=1000, type=int,
                    help='path to save models')

# modelarts modification
parser.add_argument('--train_url',
                    default='obs://mindx-user/csl-shi/wideresnet101/result/',
                    type=str,
                    help="setting dir of training output")
parser.add_argument('--data_url',
                    default='obs://mindx-user/csl-shi/wideresnet101/data/',
                    type=str,
                    help='path to dataset')

parser.add_argument('--model_url',
                    metavar='DIR',
                    default='',
                    help='path to pretrained model')
parser.add_argument('--onnx', default=True, action='store_true',
                    help="convert pth model to onnx")

cur_step = 0
CACHE_TRAINING_URL = "/cache/training/"
CACHE_DATA_URL = "/cache/data_url"
CACHE_MODEL_URL = "/cache/model"

best_acc1 = 0


def main():
    args = parser.parse_args()
    global CALCULATE_DEVICE
    CALCULATE_DEVICE = "npu:{}".format(args.npu)
    if 'npu' in CALCULATE_DEVICE:
       torch.npu.set_device(CALCULATE_DEVICE)
    if args.data_url:
        import moxing as mox
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

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    ###### modify npu_p1 1######
    args.gpu = None
    ###### modify npu_p1 1 end ######
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        ###### modify 8 ######
        if args.device == 'npu':
            dist.init_process_group(backend=args.dist_backend,  # init_method=args.dist_url,
                                    world_size=args.world_size, rank=args.rank)
        else:
            dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                    world_size=args.world_size, rank=args.rank)
        ###### modify 8 end ######
    # create model
    if args.pretrained:
        print("=> using pre-trained model wide_resnet101_2")
        model = senet.se_resnet50()
        print("loading model of yours...")
        model_path = "./checkpoint.pth.tar"
        if args.model_url:
            real_path = CACHE_MODEL_URL
            if not os.path.exists(real_path):
                os.makedirs(real_path)
            mox.file.copy_parallel(args.model_url, real_path)
            print("training data finish copy to %s." % real_path)
            model_path = os.path.join(CACHE_MODEL_URL, 'checkpoint.pth.tar')
        pretrained_dict = torch.load(model_path, map_location="cpu")["state_dict"]
        model.load_state_dict({k.replace('module.', ''): v for k, v in pretrained_dict.items()})
        if "fc.weight" in pretrained_dict:
            pretrained_dict.pop('fc.weight')
            pretrained_dict.pop('fc.bias')
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(2048, args.num_classes)
        #model.load_state_dict(pretrained_dict, strict=False)
    else:
        print("=> creating model wide_resnet101_2")
        model = senet.se_resnet50()
    ###### modify npu_p1 2######
    if args.distributed:
    ###### modify npu_p1 2 end ######
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            # model = torch.nn.DataParallel(model).cuda()
            ###### modify npu_p1 3######
            model = model.to(CALCULATE_DEVICE)
            ###### modify npu_p1 3 end ######

    # define loss function (criterion) and optimizer
    # criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    ############## npu modify 4 begin #############
    # 将损失函数迁移到NPU上进行计算。
    criterion = nn.CrossEntropyLoss().to(CALCULATE_DEVICE)
    ############## npu modify 4 end #############
    optimizer = apex.optimizers.NpuFusedSGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                nesterov=True,
                                weight_decay=args.weight_decay)
    ###### modify 1 ######
    if args.amp:
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.opt_level, loss_scale=args.loss_scale)
    ###### modify 1 end ######
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    if args.data_url:
        real_path = CACHE_DATA_URL
        if not os.path.exists(real_path):
            os.makedirs(real_path)
        mox.file.copy_parallel(args.data_url, real_path)
        print("training data finish copy to %s." % real_path)
        args.data = real_path

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
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    ###### modify 7 ######
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(
                train_sampler is None),
        num_workers=args.workers, pin_memory=False, sampler=train_sampler, drop_last=True)
    ###### modify 7 end #######
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return
    ###### modify 3 ######
    if args.prof:
        profiling(train_loader, model, criterion, optimizer, args)
        return
    ###### modify 3 end ######

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, ngpus_per_node)

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
            }, is_best)
    if args.train_url:
        mox.file.copy_parallel(CACHE_TRAINING_URL, args.train_url)


def proc_node_module(checkpoint, AttrName):
    new_state_dict = OrderedDict()
    for k, v in checkpoint[AttrName].items():
        if(k[0:7] == "module."):
            name = k[7:]
        else:
            name = k[0:]
        new_state_dict[name] = v
    return new_state_dict

def convert(model_path, onnx_save, num_class):
    checkpoint = torch.load(model_path, map_location='cpu')
    checkpoint['state_dict'] = proc_node_module(checkpoint, 'state_dict')
    model = senet.se_resnet50()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    input_names = ["actual_input_1"]
    output_names = ["output1"]
    dummy_input = torch.randn(1, 3, 224, 224)
    if len(onnx_save) > 0:
        save_path = os.path.join(onnx_save, "se_resnet50_2_npu_16.onnx")
    else:
        save_path = "se_resnet50_2_npu_16.onnx"
    print(save_path)
    torch.onnx.export(model, dummy_input, save_path
                     , input_names=input_names, output_names=output_names
                     , opset_version=11)


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
        if args.device == 'npu':
            loc = CALCULATE_DEVICE
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
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)

        if 'npu' in CALCULATE_DEVICE:
            target = target.to(torch.int32)
        images, target = images.to(CALCULATE_DEVICE, non_blocking=True), target.to(CALCULATE_DEVICE, non_blocking=True)
        ############## npu modify 5 end #############

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
        ######  modify 2 ######
        if args.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        ######  modify 2 end ######
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    ###### modify 4 ######
        if i % args.print_freq == 0:
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                          and args.rank % ngpus_per_node == 0):
                progress.display(i)

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                and args.rank % ngpus_per_node == 0):
        if batch_time.avg:
            print("[npu id:", CALCULATE_DEVICE, "]", "batch_size:", args.world_size * args.batch_size,
                  'Time: {:.3f}'.format(batch_time.avg), '* FPS@all {:.3f}'.format(
                    args.batch_size * args.world_size / batch_time.avg))
    ###### modify 4 end ######


def validate(val_loader, model, criterion, args):
    ###### modify 5 ######
    batch_time = AverageMeter('Time', ':6.3f', start_count_index= 5)
    ###### modify 5 end ######
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
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.device == 'npu':
                loc = CALCULATE_DEVICE
                images = images.to(loc).to(torch.float)
            if args.device == 'npu':
                loc = CALCULATE_DEVICE
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
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    args = parser.parse_args()
    if args.train_url:
        os.makedirs(CACHE_TRAINING_URL, 0o755, exist_ok=True)
        filename = os.path.join(CACHE_TRAINING_URL, filename)
        torch.save(state, filename)
        convert(filename, CACHE_TRAINING_URL, args.num_classes)
        path_best = os.path.join(CACHE_TRAINING_URL, 'model_best.pth.tar')
        if is_best:
            shutil.copyfile(filename, path_best)
    else:
        filename = os.path.join(args.save_path, filename)
        torch.save(state, filename)
        path_best = os.path.join(args.save_path, 'model_best.pth.tar')
        if is_best:
            shutil.copyfile(filename, path_best)


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
    """Sets the learning rate to the initial LR decayed by cosine method"""

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


if __name__ == '__main__':
    main()

