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
import glob
import numpy as np
import os
import random
import shutil
import sys
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

from apex import amp
import moxing as mox
import torch.npu

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../'))

from hook import forward_hook_fn
from hook import backward_hook_fn
from mobilenet import mobilenet_v2
from pthtar2onnx import convert


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default='/dataset/imagenet',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=128, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=600, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')
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
parser.add_argument('--dist-url', default=None, type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=49, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--amp', default=False, action='store_true',
                    help='use amp to train the model')
parser.add_argument('--opt-level', default="O2", type=str, help='apex optimize level')
parser.add_argument('--loss-scale-value', default='64', type=int, help='static loss scale value')

parser.add_argument('--summary-path', default=None, type=str, help='event file path')
parser.add_argument('--stop-step-num', default=None, type=int, help='after the stop-step, killing the training task')
parser.add_argument('--device', default='npu:0', type=str, help='device type, cpu or npu:x or cuda:x')
parser.add_argument('--eval-freq', default=5, type=int, help='test interval')
parser.add_argument('--hook', default=False, action='store_true', help='pytorch hook')
parser.add_argument('--class_nums', default=1000, type=int, help='class-nums only for pretrain')

# modelarts modification
parser.add_argument('--train_url',
                    default="/cache/training",
                    type=str,
                    help="setting dir of training output")
parser.add_argument('--data_url',
                    metavar='DIR',
                    default='/cache/data_url',
                    help='path to dataset')
parser.add_argument('--onnx', default=True, action='store_true',
                    help="convert pth model to onnx")

CACHE_MODEL_URL = "/cache/model"

best_acc1 = 0
cur_step = 0

CACHE_TRAINING_URL = "/cache/training/"

def seed_everything(seed, device):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if 'cuda' in device:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    args = parser.parse_args()

    if args.seed is not None:
        seed_everything(args.seed, args.device)

        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    main_worker(args)


def main_worker(args):
    global best_acc1
    global cur_step

    global_step = -1

    if 'npu' in args.device:
        torch.npu.set_device(args.device)
    if 'cuda' in args.device:
        torch.cuda.set_device(args.device)

    model = mobilenet_v2(num_classes=args.class_nums)

    # set hook
    if args.hook:
        modules = model.named_modules()
        for name, module in modules:
            module.register_forward_hook(forward_hook_fn)
            module.register_backward_hook(backward_hook_fn)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    criterion = nn.CrossEntropyLoss()

    if 'npu' in args.device or 'cuda' in args.device:
        model = model.to(args.device)
        criterion = criterion.to(args.device)

    if args.amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level, loss_scale=args.loss_scale_value)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=args.device)
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

    if args.pretrain:
        # ------------------modelarts modification----------------------
        os.makedirs(CACHE_MODEL_URL, exist_ok=True)
        mox.file.copy_parallel(args.pretrain, os.path.join(CACHE_MODEL_URL, "checkpoint.pth"))
        # ------------------modelarts modification---------------------
        args.pretrain = os.path.join(CACHE_MODEL_URL, "checkpoint.pth")
        if not os.path.isfile(args.pretrain):
            print("no chechpoint found at {}".format(args.pretrain))

        print("loading checkpoint '{}'".format(args.pretrain))
        pretrained_dict = torch.load(args.pretrain, map_location="cpu")['state_dict']
        pretrained_dict.pop('module.classifier.1.weight')
        pretrained_dict.pop('module.classifier.1.bias')
        model.load_state_dict(pretrained_dict, strict=False)
        print("loaded checkpoint '{}'".format(args.pretrain))

    # Data loading code
    # -------modelarts modification-------
    real_path = '/cache/data_url'
    if not os.path.exists(real_path):
        os.makedirs(real_path)
    mox.file.copy_parallel(args.data_url, real_path)
    print("training data finish copy to %s." % real_path)
    # ---------modelarts modification-----

    traindir = os.path.join(real_path, 'train')
    valdir = os.path.join(real_path, 'val')

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
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args, global_step)
        return

    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        global_step = train(train_loader, model, criterion, optimizer, epoch, args, global_step)

        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            # evaluate on validation set
            acc1 = validate(val_loader, model, criterion, args, global_step)

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            # save checkpoint
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

        if args.stop_step_num is not None and cur_step >= args.stop_step_num:
            break

    if args.onnx:
        convert_pth_to_onnx(args)

    # --------------modelarts modification----------
    mox.file.copy_parallel(CACHE_TRAINING_URL, args.train_url)
    # --------------modelarts modification end----------


def convert_pth_to_onnx(args):
    pth_pattern = os.path.join(CACHE_TRAINING_URL, 'checkpoint.pth.tar')
    pth_file_list = glob.glob(pth_pattern)
    if not pth_file_list:
        print(f"can't find pth {pth_pattern}")
        return
    pth_file = pth_file_list[0]
    onnx_path = pth_file.split(".")[0] + '.onnx'
    convert(pth_file, onnx_path, args.class_nums)


def train(train_loader, model, criterion, optimizer, epoch, args, global_step, sum_writer=None):
    global cur_step

    if args.seed is not None:
        seed_everything(args.seed + epoch, args.device)

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    learning_rate = AverageMeter('LR', ':2.8f')
    losses = AverageMeter('Loss', ':6.8f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, learning_rate, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    steps_per_epoch = len(train_loader)
    for i, (images, target) in enumerate(train_loader):

        global_step = epoch * steps_per_epoch + i
        cur_step = global_step

        lr = adjust_learning_rate(optimizer, global_step, steps_per_epoch, args)

        learning_rate.update(lr)

        # measure data loading time
        data_time.update(time.time() - end)

        if 'npu' in args.device:
            target = target.to(torch.int32)

        if 'npu' in args.device or 'cuda' in args.device:
            images = images.to(args.device, non_blocking=True)
            target = target.to(args.device, non_blocking=True)

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

        if args.stop_step_num is not None and cur_step >= args.stop_step_num:
            break

    print(' * FPS@all {:.3f}'.format(args.batch_size / (batch_time.avg + 0.001)))
    return global_step


def validate(val_loader, model, criterion, args, global_step, sum_writer=None):
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

            if 'npu' in args.device:
                target = target.to(torch.int32)

            if 'npu' in args.device or 'cuda' in args.device:
                images = images.to(args.device, non_blocking=True)
                target = target.to(args.device, non_blocking=True)

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

    if not os.path.exists(CACHE_TRAINING_URL):
        os.makedirs(CACHE_TRAINING_URL)

    checkpoint_save_path = os.path.join(CACHE_TRAINING_URL, filename)
    torch.save(state, checkpoint_save_path)
    if is_best:
        shutil.copyfile(checkpoint_save_path, os.path.join(CACHE_TRAINING_URL, "model_best.pth.tar"))


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


if __name__ == '__main__':
    main()
