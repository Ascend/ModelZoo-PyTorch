# Copyright (c) Soumith Chintala 2016,
# Copyright 2021 Huawei Technologies Co., Ltd
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


import argparse
import glob
import os
import random
import shutil
import sys
import time
import warnings

import torch
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

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../'))

# import model
from inceptionv4 import inceptionv4
from inceptionv4_v2 import inceptionv4 as inceptionv4_v2

import numpy as np
from apex import amp
# modelarts modification
import moxing as mox
from collections import OrderedDict

warnings.filterwarnings('ignore')

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--data_url',
                    metavar='DIR',
                    default='/cache/data_url',
                    help='path to dataset')
parser.add_argument('--data', metavar='DIR', default='/opt/npu/dataset/imagenet',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='inceptionv4',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=20, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=384, type=int,
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
parser.add_argument('-ef', '--eval-freq', dest='eval_freq', default=5, type=int,
                    metavar='N', help='evaluate frequency (default: 5)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--fine_tuning', dest='fine_tuning', action='store_true',
                    help='use fine-tuning model')
parser.add_argument('--pretrained_weight', dest='pretrained_weight',
                    help='pretrained weight dir')             
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
parser.add_argument('--multiprocessing-distributed', dest='multiprocessing_distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

## for ascend 910
parser.add_argument('--npu', default=0, type=int,
                    help='NPU id to use.')
parser.add_argument('--device', default='npu', type=str, help='npu or gpu')
parser.add_argument('--addr', default='10.136.181.115',
                    type=str, help='master addr')
parser.add_argument('--device_list', default='0,1,2,3,4,5,6,7',
                    type=str, help='device id list')
parser.add_argument('--warm_up_epochs', default=0, type=int,
                    help='warm up')
parser.add_argument('--amp', default=False, action='store_true',
                    help='use amp to train the model')
parser.add_argument('--loss-scale', default=1024., type=float,
                    help='loss scale using in amp, default -1 means dynamic')
parser.add_argument('--opt-level', default='O2', type=str,
                    help='loss scale using in amp, default -1 means dynamic')
parser.add_argument('--prof', default=False, action='store_true',
                    help='use profiling to evaluate the performance of model')
parser.add_argument('--label-smoothing',
                    default=0.0,
                    type=float,
                    metavar='S',
                    help='label smoothing')


# modelarts
parser.add_argument('--train_url',
                    default="/cache/training",
                    type=str,
                    help="setting dir of training output")
parser.add_argument('--onnx', default=True, action='store_true',
                    help="convert pth model to onnx")
parser.add_argument('--class_num', default=1000, type=int,
                    help='number of class')

best_acc1 = 0
cur_step = 0
CACHE_TRAINING_URL = "/cache/training"

def main():
    args = parser.parse_args()
    print("===================main===================")
    print(args)
    print("===================main===================")

    if args.npu is None:
        args.npu = 0
    global CALCULATE_DEVICE
    global best_acc1

    #-----modelarts modification-----------------
    CALCULATE_DEVICE = "npu:{}".format(args.npu)
    #-----modelarts modification-----------------
    torch.npu.set_device(CALCULATE_DEVICE)

    if args.seed is not None:
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.npu.device_count()
    print('{} node found.'.format(ngpus_per_node))
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
    global cur_step
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

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
        CACHE_MODEL_URL = "/cache/model"
        # ------------------modelarts modification----------------------
        os.makedirs(CACHE_MODEL_URL, exist_ok=True)
        mox.file.copy_parallel(args.pretrained_weight, os.path.join(CACHE_MODEL_URL, "checkpoint.pth.tar"))
        # ------------------modelarts modification---------------------
        args.pretrained_weight = os.path.join(CACHE_MODEL_URL, "checkpoint.pth.tar")
        model = inceptionv4(pretrained=False)
        
        checkpoint = torch.load(args.pretrained_weight, map_location='cpu')
        
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items() if 'last_linear' not in k}, strict=False)

    else:
        print("=> creating model '{}'".format(args.arch))
        model = inceptionv4(num_classes=args.class_num, pretrained=False)

    parameters = model.parameters()
    if args.fine_tuning:
        print("=> transfer-learning mode + fine-tuning (train only the last FC layer)")
        for param in model.parameters():
            param.requires_grad = False
        if args.arch == 'inceptionv4':
            model.last_linear = nn.Linear(1536, args.class_num)
            parameters = model.last_linear.parameters()
        elif args.arch == 'inceptionv4_v2':
            model.last_linear = nn.Linear(1536, args.class_num)
            parameters = model.last_linear.parameters(args.class_num)
        else:
            print("Error:Fine-tuning is not supported on this architecture")
            exit(-1)
    else:
        parameters = model.parameters()

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
            model = model.to(CALCULATE_DEVICE)
            print("before : model = torch.nn.DataParallel(model).cuda()") 

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(CALCULATE_DEVICE)

    optimizer = torch.optim.SGD(parameters, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level, loss_scale=args.loss_scale)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=CALCULATE_DEVICE)
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

    cudnn.benchmark = True

    # -------modelarts modification-------
    real_path = '/cache/data_url'
    if not os.path.exists(real_path):
        os.makedirs(real_path)
    mox.file.copy_parallel(args.data_url, real_path)
    print("training data finish copy to %s." % real_path)
    # ---------modelarts modification-----

    # Data loading code
    traindir = os.path.join(real_path, 'train')
    valdir = os.path.join(real_path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.Resize([299,299]),
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
        num_workers=args.workers, pin_memory=False, sampler=train_sampler, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize([299,299]),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False, drop_last=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        if (epoch + 1) % (args.eval_freq) == 0 or epoch == args.epochs - 1:
            # evaluate on validation set
            acc1 = validate(val_loader, model, criterion, args, epoch)

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                        and args.rank % ngpus_per_node == 0
                                                        and epoch == args.epochs - 1):
                if args.amp:
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'best_acc1': best_acc1,
                        'optimizer': optimizer.state_dict(),
                        'amp': amp.state_dict(),
                    }, is_best)
                else:
                    # print('==save==')
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'best_acc1': best_acc1,
                        'optimizer': optimizer.state_dict(),
                    }, is_best, args)

    if args.onnx:
        convert_pth_to_onnx(args)

    # --------------modelarts modification----------
    mox.file.copy_parallel(CACHE_TRAINING_URL, args.train_url)
    # --------------modelarts modification end----------

def proc_node_module(checkpoint, attr_name):
    """
    modify state_dict
    :param checkpoint: loaded model file
    :param attr_name: key state_dict
    :return: new state_dict
    """
    new_state_dict = OrderedDict()
    for k, v in checkpoint[attr_name].items():
        if k[0:7] == "module.":
            name = k[7:]
        else:
            name = k[0:]
        new_state_dict[name] = v
    return new_state_dict

def convert(pth_file_path, onnx_file_path, class_nums):
    checkpoint = torch.load(pth_file_path, map_location='cpu')
    checkpoint['state_dict'] = proc_node_module(checkpoint, 'state_dict')

    model = inceptionv4_v2(pretrained=False, num_classes=class_nums)
    model.load_state_dict(checkpoint['state_dict'],False)
    model.eval()

    input_names = ["actual_input_1"]
    output_names = ["output1"]
    dummy_input = torch.randn(32, 3, 299, 299)
    torch.onnx.export(model, dummy_input, onnx_file_path, input_names=input_names, output_names=output_names, opset_version=11)

def convert_pth_to_onnx(config_args):
    pth_pattern = os.path.join(CACHE_TRAINING_URL, 'checkpoint.pth.tar')
    pth_file_list = glob.glob(pth_pattern)
    if not pth_file_list:
        print(f"can't find pth {pth_pattern}")
        return
    pth_file = pth_file_list[0]
    onnx_path = pth_file.split(".")[0] + '.onnx'
    convert(pth_file, onnx_path, config_args.class_num)


def train(train_loader, model, criterion, optimizer, epoch, args):
    global cur_step
    batch_time = AverageMeter('Time', ':6.3f', start_count_index=0)
    data_time = AverageMeter('Data', ':6.3f', start_count_index=0)
    losses = AverageMeter('Loss', ':.4e', start_count_index=0)
    top1 = AverageMeter('Acc@1', ':6.2f', start_count_index=0)
    top5 = AverageMeter('Acc@5', ':6.2f', start_count_index=0)
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    step_per_epoch = len(train_loader)
    for i, (images, target) in enumerate(train_loader):
        cur_step = epoch * step_per_epoch + i
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.to(torch.int32)
        images, target = images.to(CALCULATE_DEVICE, non_blocking=False), target.to(CALCULATE_DEVICE, non_blocking=False)
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
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    print(' * FPS@all {:.3f}'.format(args.batch_size / (batch_time.avg+0.001)))


def validate(val_loader, model, criterion, args, epoch=0, writer=None):
    print('==validate==')
    batch_time = AverageMeter('Time', ':6.3f', start_count_index=0)
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
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            target = target.to(torch.int32)
            images, target = images.to(CALCULATE_DEVICE, non_blocking=False), target.to(CALCULATE_DEVICE,
                                                                                        non_blocking=False)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            print('acc1={}, acc5={}, images_size(0)={}'.format(acc1[0], acc5[0], images.size(0)))
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


def save_checkpoint(state, is_best, args, filename='checkpoint.pth.tar'):
    if not os.path.exists(CACHE_TRAINING_URL):
        os.makedirs(CACHE_TRAINING_URL, 0o755)

    checkpoint_save_path = os.path.join(CACHE_TRAINING_URL, filename)
    torch.save(state, checkpoint_save_path)
    if is_best:
        
        # --------------modelarts modification----------
        mox.file.copy_parallel(checkpoint_save_path, args.train_url + 'model_best_acc%.4f_epoch%d.pth.tar' % (state['best_acc1'], state['epoch']))
        # --------------modelarts modification end----------


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
        self.index = 0

    def update(self, val, n=1):
        self.val = val
        self.count += n
        self.index += 1
        if self.index > self.start_count_index:
            self.sum += val * n
            self.avg = self.sum / self.count

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


if __name__ == '__main__':
    main()
