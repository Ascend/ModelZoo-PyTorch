# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
# Copyright 2020 Huawei Technologies Co., Ltd
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
import torch.npu
import DistributedResnet50.image_classification.resnet as nvmodels
from apex import amp

BATCH_SIZE = 512
EPOCHS_SIZE = 100
TRAIN_STEP = 8000
LOG_STEP = 100

CALCULATE_DEVICE = "npu:7"
PRINT_DEVICE = "cpu"
SOURCE_DIR = "/data/imagenet"

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data',
                    metavar='DIR',
                    default=SOURCE_DIR,
                    help='path to dataset')
parser.add_argument('--save_ckpt_path',
                    metavar='DIR',
                    default='./',
                    help='path of checkpoint file')
parser.add_argument('-a', '--arch',
                    metavar='ARCH',
                    default='resnet50',
                    choices=model_names,
                    help='default: resnet50')
parser.add_argument('-j', '--workers',
                    default=32,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--num_classes',
                    default=1000,
                    type=int,
                    metavar='N',
                    help='class number of dataset')
parser.add_argument('--epochs',
                    default=EPOCHS_SIZE,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size',
                    default=BATCH_SIZE,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate',
                    default=0.1,
                    type=float,
                    metavar='LR',
                    help='initial learning rate',
                    dest='lr')
parser.add_argument('--momentum',
                    default=0.9,
                    type=float,
                    metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay',
                    default=1e-4,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq',
                    default=10,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--resume',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate',
                    dest='evaluate',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained',
                    dest='pretrained',
                    action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size',
                    default=-1,
                    type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank',
                    default=-1,
                    type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url',
                    default='tcp://224.66.41.62:23456',
                    type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend',
                    default='nccl',
                    type=str,
                    help='distributed backend')
parser.add_argument('--seed',
                    default=None,
                    type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu',
                    default=None,
                    type=int,
                    help='GPU id to use.')
parser.add_argument('--npu',
                    default=None,
                    type=int,
                    help='NPU id to use.')
parser.add_argument('--multiprocessing-distributed',
                    action='store_true')
parser.add_argument('--warmup',
                    default=0,
                    type=int,
                    metavar='E',
                    help='number of warmup epochs')
parser.add_argument('--label-smoothing',
                    default=0.0,
                    type=float,
                    metavar='S',
                    help='label smoothing')
parser.add_argument('--optimizer-batch-size',
                    default=-1,
                    type=int,
                    metavar='N',
                    help=
                    'size of a total batch size, for simulating bigger batches using gradient accumulation')
parser.add_argument('--static-loss-scale',
                    type=float,
                    default=1,
                    help=
                    'Static loss scale, positive power of 2 values can improve fp16 convergence.')
parser.add_argument('-t',
                    '--fine-tuning',
                    action='store_true',
                    help='transfer learning + fine tuning - train only the last FC layer.')
parser.add_argument('--precision_mode',
                    default='allow_mix_precision',
                    type=str,
                    help='precision_mode')
# 图模式
parser.add_argument('--graph_mode',
                    action='store_true',
                    help='whether to enable graph mode.')
best_acc1 = 0
args = parser.parse_args()
def main():
    
    if args.npu is None:
        args.npu = 0
    global CALCULATE_DEVICE
    CALCULATE_DEVICE = "npu:{}".format(args.npu)
    torch.npu.set_device(CALCULATE_DEVICE)
    print("use ", CALCULATE_DEVICE)

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
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.precision_mode == "must_keep_origin_dtype":
        option = {}
        option["ACL_PRECISION_MODE"] = "must_keep_origin_dtype" 
        torch.npu.set_option(option) 

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = nvmodels.build_resnet("resnet50", "classic", True)
        print("load pretrained model")
        pretrained_dict = \
        torch.load("/home/checkpoint_npu0model_best.pth.tar", map_location="cpu")["state_dict"]
        pretrained_dict.pop('module.fc.weight')
        pretrained_dict.pop('module.fc.bias')
        model.load_state_dict(pretrained_dict, strict=False)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](zero_init_residual=True, num_classes=args.num_classes)

    if args.fine_tuning:
        print("=> transfer learning + fine tuning(train only the last FC layer)")
        if args.arch == "resnet50":
            model.parameters()
        else:
            print("Error: Fine-tuning is not supported on this architecture")
            exit(-1)
    else:
        model.parameters()

    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            torch.nn.init.kaiming_normal_(layer.weight, a=math.sqrt(5), )
    if args.distributed:
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
            model = model.to(CALCULATE_DEVICE)

    lr_policy = lr_cosine_policy(args.lr,
                                 args.warmup,
                                 args.epochs)


    # define loss function (criterion) and optimizer
    loss = nn.CrossEntropyLoss
    if args.label_smoothing > 0.0:
        loss = lambda: LabelSmoothing(args.label_smoothing)
    criterion = loss().to(CALCULATE_DEVICE)
    optimizer = torch.optim.SGD([
        {'params': [param for name, param in model.named_parameters() if name[-4:] == 'bias'], 'weight_decay': 0.0},
        {'params': [param for name, param in model.named_parameters() if name[-4:] != 'bias'], 'weight_decay': args.weight_decay}],
                                args.lr,
                                momentum=args.momentum)  # torch.optim.  apex.optimizers.NpuFusedSGD
    if args.precision_mode == "must_keep_origin_dtype":
        model, optimizer = amp.initialize(model, optimizer, opt_level="O0", verbosity=1,combine_grad=False)
    else:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O2", loss_scale=1024, verbosity=1,combine_grad=False)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.npu is not None:
                checkpoint = torch.load(args.resume)
            elif args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.npu is not None:
                best_acc1 = best_acc1.to("npu:{}".format(args.npu))
            elif args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

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

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        lr_policy(optimizer, 0, epoch)
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        file_name = "checkpoint_npu{}".format(args.npu)
        modeltmp = model.cpu()
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': modeltmp.state_dict(),
            'best_acc1': best_acc1,
        }, is_best, args, file_name)
        modeltmp.to(CALCULATE_DEVICE)

def train(train_loader, model, criterion, optimizer, epoch, args):
    if args.optimizer_batch_size < 0:
        batch_size_multiplier = 1
    else:
        tbs = 1 * args.batch_size
        if args.optimizer_batch_size % tbs != 0:
            print(
                "Warning: simulated batch size {} is not divisible by actual batch size {}"
                    .format(args.optimizer_batch_size, tbs))
        batch_size_multiplier = int(args.optimizer_batch_size / tbs)
        print("BSM: {}".format(batch_size_multiplier))

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
    optimizer.zero_grad()
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # 图模式
        if args.graph_mode:
            print("args.graph_mode")
            torch.npu.enable_graph_mode()

        if i > 1000:
            pass
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)

        images = images.to(CALCULATE_DEVICE, non_blocking=True)
        if args.label_smoothing == 0.0:
        # 图模式
            if args.graph_mode:
                print("args.graph_mode")
                target = target.to(CALCULATE_DEVICE, non_blocking=True).to(torch.int32)
            else:
                target = target.to(torch.int32).to(CALCULATE_DEVICE, non_blocking=True)
        # compute output
        output = model(images)
        loss = criterion(output, target)

        if args.label_smoothing > 0.0:
        # 图模式
            if args.graph_mode:
                print("args.graph_mode")
                target = target.to(CALCULATE_DEVICE, non_blocking=True).to(torch.int32)
            else:
                target = target.to(torch.int32).to(CALCULATE_DEVICE, non_blocking=True)
        
        

        # measure accuracy and record loss
        # 图模式
        if not args.graph_mode:
            # print("args.graph_mode====================")
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer_step = ((i + 1) % batch_size_multiplier) == 0
        if optimizer_step:
            if batch_size_multiplier != 1:
                for param_group in optimizer.param_groups:
                    for param in param_group['params']:
                        param.grad /= batch_size_multiplier
            optimizer.step()
            optimizer.zero_grad()
        
        # 图模式
        if args.graph_mode:
            print("args.graph_mode")
            torch.npu.launch_graph()
            if i == 100:
                torch.npu.synchronize()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i < 11 or i % LOG_STEP == 0:
            progress.display(i)

        if i == TRAIN_STEP:
            break
    # 图模式
    if args.graph_mode:
        print("args.graph_mode")
        torch.npu.disable_graph_mode()

    if batch_time.avg > 0:
        print("batch_size:", args.batch_size, 'Time: {:.3f}'.format(batch_time.avg), '* FPS@all {:.3f}'.format(
                args.batch_size/batch_time.avg))

def validate(val_loader, model, criterion, args):
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
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if i > 50:
                pass
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            images = images.to(CALCULATE_DEVICE, non_blocking=True)
            if args.label_smoothing == 0.0:
                target = target.to(torch.int32).to(CALCULATE_DEVICE, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            if args.label_smoothing > 0.0:
                target = target.to(torch.int32).to(CALCULATE_DEVICE, non_blocking=True)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % LOG_STEP == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
    return top1.avg

def save_checkpoint(state, is_best, args, filename='checkpoint'):
    filename2 = os.path.join(args.save_ckpt_path, filename + ".pth.tar")
    torch.save(state, filename2)
    if is_best:
        shutil.copyfile(filename2, os.path.join(args.save_ckpt_path, filename+'model_best.pth.tar'))

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

class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.

        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1).to(CALCULATE_DEVICE))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

def lr_policy(lr_fn, logger=None):
    if logger is not None:
        logger.register_metric('lr',
                               log.LR_METER(),
                               verbosity=dllogger.Verbosity.VERBOSE)

    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)

        if logger is not None:
            logger.log_metric('lr', lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return _alr

def lr_cosine_policy(base_lr, warmup_length, epochs, logger=None):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr

    return lr_policy(_lr_fn, logger=logger)

if __name__ == '__main__':
    main()
