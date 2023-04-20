# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================
import argparse
import os
import shutil
import time
import random

import torch
if torch.__version__ >= "1.8":
    import torch_npu
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.distributed as dist
import mobilenetv3
import apex
from apex import amp


from auto_augment import rand_augment_transform, augment_and_mix_transform, auto_augment_transform
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


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

model_names.append('mobilenet')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default = "/home/jcwang/dataset/imagenet-data",
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='mobilenet',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=64, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--max_steps', default=None, type=int, metavar='N',
                    help='number of total steps to run')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    default = 0,
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')

# Mixed precision training parameters
parser.add_argument('--apex', action='store_true',
                    help='Use apex for mixed precision training')
parser.add_argument('--apex-opt-level', default='O2', type=str,
                    help='For apex mixed precision training'
                         'O0 for FP32 training, O1 for mixed precision training.'
                         'For further detail, see https://github.com/NVIDIA/apex/tree/master/examples/imagenet')
parser.add_argument('--loss-scale-value', default=1024., type=float,
                    help='loss scale using in amp, default -1 means dynamic')
parser.add_argument('--seed',
                    default=1,
                    type=int,
                    help='Manually set random seed')
## for ascend 910
parser.add_argument('--device_id', default=5, type=int, help='device id')


# distributed training parameters
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='env://',
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='hccl', type=str,
                    help='distributed backend')
parser.add_argument('--dist-rank',
                    default=0,
                    type=int,
                    help='node rank for distributed training')
parser.add_argument('--distributed',
                    action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs.')
#vision						 
parser.add_argument('--lr-step-size', default=30, type=int, help='decrease lr every step-size epochs')
parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')

best_prec1 = 0

def mobilenet(path="./checkpoint.pth.tar"):
    net = Net()
    state_dict = torch.load(path)
    net.load_state_dict(state_dict)
    return net


def main():
    global best_prec1
    args = parser.parse_args()

    os.environ['MASTER_ADDR'] = str(os.getenv('MASTER_ADDR','127.0.0.1'))
    os.environ['MASTER_PORT'] = '29688'

    #设置seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    if args.distributed:
        if 'RANK_SIZE' in os.environ and 'RANK_ID' in os.environ:
            args.rank_size = int(os.environ['RANK_SIZE'])
            args.rank_id = int(os.environ['RANK_ID'])
            if args.rank_size <= 8:
                args.rank = args.dist_rank * args.rank_size + args.rank_id
            else:
                args.rank = int(os.environ['RANK'])
            args.world_size = args.world_size * args.rank_size
            args.device_id = args.rank_id
            args.batch_size = int(args.batch_size / args.rank_size)
            args.workers = int((args.workers + args.rank_size - 1) / args.rank_size)
            dist.init_process_group(backend=args.dist_backend,  # init_method=args.dist_url,
                                    world_size=args.world_size, rank=args.rank)
        else:
            raise RuntimeError("Please set RANK_SIZE and RANK_ID for .")

    if args.distributed:
        print("Use NPU: {} for training".format(args.rank_id))
    else:
        print("Use NPU: {} for training".format(args.device_id))

    print(args)

    device = torch.device(f'npu:{args.device_id}')
    torch.npu.set_device(device)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        if args.arch.startswith('mobilenet'):
            model = mobilenetv3.mobilenet_v3_large()
        else:
            model = models.__dict__[args.arch]()

    model = model.to(device)


    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)

    # vision optimizer
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, momentum=args.momentum,
                                    #weight_decay=args.weight_decay, eps=0.0316, alpha=0.9)
    # prepare for new version, significant improvement
    optimizer = apex.optimizers.NpuFusedRMSprop(model.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay, eps=0.0316, alpha=0.9)


    if args.apex:
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level='O2',
                                          loss_scale=args.loss_scale_value,
                                          combine_grad=True)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.device_id])

   # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
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

    train_dataset = datasets.ImageFolder(traindir, transforms.Compose([
        transforms.RandomSizedCrop(224),
        auto_augment_wrapper(224),
        transforms.ToTensor(),
        normalize,
    ]))
	
    # vision lr
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
	
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)


    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False, drop_last=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return


    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed:
            train_sampler.set_epoch(epoch)

        lr_scheduler.step()
		
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, device, args.apex)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, device, args)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, args, device, use_apex=False):
    batch_time = AverageMeter()
    img_per_s = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    profile = Profile(start_step=int(os.getenv('PROFILE_START_STEP', 10)),
                      profile_type=os.getenv('PROFILE_TYPE'))
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.to(device)
        images = images.to(device,non_blocking=True)
        input_var = torch.autograd.Variable(images)
        target_var = torch.autograd.Variable(target)

        profile.start()
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(prec1.item(), images.size(0))
        top5.update(prec5.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if use_apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        profile.end()

        # measure elapsed time
        batch_time.update(time.time() - end)
        img_per_s.update((args.world_size * args.batch_size) / (time.time() - end))
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'img/s {img_per_s.val:.3f}\t'
                  'Avg {img_per_s.avg:.3f}\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time, img_per_s=img_per_s,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
        if args.max_steps and i > args.max_steps:
            break


def validate(val_loader, model, criterion, device, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    for i, (images, target) in enumerate(val_loader):
        target = target.to(device)
        input_var  = images.to(device)
        target_var = target

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(prec1.item(), images.size(0))
        top5.update(prec5.item(), images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate_fraction_epoch(optimizer, epoch, args):
    """Use the epoch cosine schedule"""

    alpha = 0
    cosine_decay = 0.5 * (1 + np.cos(np.pi * epoch / args.epochs))
    decayed = (1 - alpha) * cosine_decay + alpha
    lr = args.lr * decayed
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
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

if __name__ == '__main__':
    main()
