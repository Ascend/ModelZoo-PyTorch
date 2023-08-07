# BSD 3-Clause License

# Copyright (c) Soumith Chintala 2016,
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

import argparse
import os
import random
import shutil
import time
import warnings

import torch
if torch.__version__ >= '1.8':
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
from inception import inception_v3
try:
    from torch_npu.utils.profiler import Profile
except ImportError:
    print("Profile not in torch_npu.utils.profiler now.. Auto Profile disabled.", flush=True)
    class Profile:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def end(self):
            pass

from apex import amp
import apex
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default='/opt/npu/dataset/imagenet',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=512, type=int,
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
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--fine-tuning', action='store_true',
                    help='use fine-tuning model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='', type=str,
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
parser.add_argument('--device', default='npu', type=str,
                    help='npu or gpu')
parser.add_argument('--addr', default='', type=str,
                    help='master addr')
parser.add_argument('--checkpoint-nameprefix', default='checkpoint', type=str,
                    help='checkpoint-nameprefix')
parser.add_argument('--checkpoint-freq', default=0, type=int,
                    metavar='N', help='checkpoint frequency (default: 0)'
                                      '0: save only one file whitch per epoch;'
                                      'n: save diff file per n epoch'
                                      '-1:no checkpoint,not support')
parser.add_argument('--device_list', default='0,1,2,3,4,5,6,7', type=str, help='device id list')
# apex
parser.add_argument('--amp', default=False, action='store_true',
                    help='use amp to train the model')
parser.add_argument('--loss_scale', default=1024,
                    help='loss scale using in amp, default -1 means dynamic')
parser.add_argument('--opt-level', default='O2', type=str,
                    help='loss scale using in amp, default -1 means dynamic')

parser.add_argument('--label-smoothing',
                    default=0.0,
                    type=float,
                    metavar='S',
                    help='label smoothing')
parser.add_argument('--warm_up_epochs', default=0, type=int,
                    help='warm up')


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
    os.environ['MASTER_PORT'] = '29688'

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    
    args.process_device_map = device_id_to_process_device_map(args.device_list)

    if args.distributed:
        if args.device == 'npu':
            ngpus_per_node = len(args.process_device_map)
        else:
            ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        # The child process uses the environment variables of the parent process,
        # we have to set KERNEL_NAME_ID for every proc
        main_worker(args.gpu, ngpus_per_node, args)

    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    if args.distributed:
        args.gpu = args.process_device_map[gpu]
    else:
        args.gpu = gpu

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

    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        #model = inception_v3(pretrained=True)
        model = inception_v3()
        print("Load my train models...")
        pretrained_dict = \
        torch.load("/home/Inception/model_best.pthtar", map_location="cpu")["state_dict"]
        model.load_state_dict(pretrained_dict, strict=False)         
    else:
        print("=> creating model '{}'".format(args.arch))
        model = inception_v3()

    if args.fine_tuning:
        print("=> transfer-learning mode + fine-tuning (train only the last FC layer)")
        for param in model.parameters():
            param.requires_grad = False
        if args.arch == 'inception_v3':
            model.classifier = nn.Linear(1024, 1000)
            parameters = model.classifier.parameters()
        else:
            print("Error:Fine-tuning is not supported on this architecture")
            exit(-1)
    else:
        parameters = model.parameters()

    args.batch_size = int(args.batch_size / ngpus_per_node)
    args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(299),
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
            transforms.Resize(342),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False, drop_last=True)

    # create model
    model = model.to(loc)

    # define loss function (criterion) and optimizer

    loss = nn.CrossEntropyLoss().to(loc)
    if args.label_smoothing > 0.0:        
        loss = lambda: LabelSmoothing(loc, args.label_smoothing)
    criterion = loss().to(loc)
    
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level, loss_scale=args.loss_scale)
    if args.distributed:
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
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    if args.evaluate:
        validate(val_loader, model, criterion, args, ngpus_per_node)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        lr = adjust_learning_rate(optimizer, epoch, args)
        
        steps_per_epoch = len(train_loader)
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
            and args.rank % ngpus_per_node == 0):
            print("=> Epoch[%d] Setting lr: %.4f" % (epoch, lr))

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, ngpus_per_node)
        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args, ngpus_per_node)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
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
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                    }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, args, ngpus_per_node):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    fps = AverageMeter('FPS', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5, fps],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    end = time.time()
    loc = 'npu:{}'.format(args.gpu)

    # steps_per_epoch = len(train_loader)
    steps_per_epoch = len(train_loader)
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
            and args.rank % ngpus_per_node == 0):
        print('==========step per epoch======================', steps_per_epoch)

    profile = Profile(start_step=int(os.getenv('PROFILE_START_STEP', 10)),
                      profile_type=os.getenv('PROFILE_TYPE'))

    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        start_time = time.time()
        target = target.to(torch.int32)
        images, target = images.to(loc, non_blocking=False), target.to(loc, non_blocking=False)

        profile.start()
        # compute output
        loss,output = get_loss(model, target, images, criterion)
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))


        if args.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        profile.end()
        fps.update(args.batch_size/(time.time() - end))

        if i % args.print_freq == 0:
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                        and args.rank % ngpus_per_node == 0):
                progress.display(i)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i < 2:
            print("step_time = %.4f" % (time.time() - start_time), flush=True)

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
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
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

            if i % args.print_freq == 0:
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                            and args.rank % ngpus_per_node == 0):
                    progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            print("[npu id:", args.gpu, "]", '[AVG-ACC] * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))

    return top1.avg

def get_loss(model, target, images, criterion):
     output,aux1 = model(images)
     loss1 = criterion(output, target)
     loss2 = criterion(aux1, target)
     # According to the paper BN auxiliary classifier
     loss = loss1 + 0.4*loss2
     return loss, output


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
        self.val = val
        self.count += n
        if self.count > (self.start_count_index * n):
            self.sum += val * n
            self.avg = self.sum / (self.count - self.start_count_index * n)

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

if __name__ == '__main__':
    main()
