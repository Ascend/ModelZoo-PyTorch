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
import random
import shutil
import time
import warnings

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
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import AverageMeter, accuracy, ProgressMeter
from utils import get_default_ImageNet_val_loader, get_default_ImageNet_train_sampler_loader, log_msg

from repvgg import get_RepVGG_func_by_name
from apex import amp
from apex.optimizers import NpuFusedSGD
try:
    from torch_npu.utils.profiler import Profile
except:
    print("Profile not in torch_npu.utils.profiler now..Auto Profile disabled.", flush=True)
    class Profile:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def end(self):
            pass

IMAGENET_TRAINSET_SIZE = 1281167


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# base settings
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='RepVGG-A0')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

# training settings
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--val-batch-size', default=100, type=int, metavar='V',
                    help='validation batch size')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--custom-weight-decay', dest='custom_weight_decay', action='store_true',
                    help='Use custom weight decay. It improves the accuracy and makes quantization easier.')

# distributed settings
parser.add_argument("--device", default="npu", type=str,
                    help="the device of training")
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--addr', default='', type=str, help='master addr')
parser.add_argument('--port', default='', type=str, help='master port')
parser.add_argument("--rank_id", dest="rank_id", default=0, type=int)
parser.add_argument("--num_gpus", default=1, type=int)

# apex setting
parser.add_argument('--amp', default=False, action='store_true',
                    help='use amp to train the model')
parser.add_argument('--opt-level', default=None, type=str, help='apex optimize level')
parser.add_argument('--loss-scale-value', default='dynamic', type=str, help='static loss scale value')

# other settings
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--tag', default='testtest', type=str,
                    help='the tag for identifying the log and model files. Just a string.')
parser.add_argument('--profile', default=0, type=int, help="profile flag")

#finetune setting
parser.add_argument('--finetune', default=0, type=int, help="profile flag")
parser.add_argument('--fclasses', default=1000, type=int, help="new dataset class number")
parser.add_argument('--fresume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

best_acc1 = 0

def sgd_optimizer(model, lr, momentum, weight_decay, use_custwd):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        apply_weight_decay = weight_decay
        apply_lr = lr
        if (use_custwd and ('rbr_dense' in key or 'rbr_1x1' in key)) or 'bias' in key or 'bn' in key:
            apply_weight_decay = 0
        if 'bias' in key:
            # Just a Caffe-style common practice. Made no difference.
            apply_lr = 2 * lr 
        params += [{'params': [value], 'lr': apply_lr, 'weight_decay': apply_weight_decay}]
        
    params_dict = {}
    for param in params:
        p, l , w = param["params"], param["lr"], param["weight_decay"] 
        k = "{}_{}".format(l, w)
        
        if k not in params_dict:
            params_dict[k] = []
        params_dict[k].append(p[0])

    params = []
    for k in params_dict:
        lr, weight_decay = map(float, k.split("_"))
        params += [{"params": params_dict[k], "lr": lr, "weight_decay": weight_decay}]
        
    optimizer = NpuFusedSGD(params, lr, momentum=momentum)
    return optimizer

# custom function to load model when not all dict elements
def load_my_pretrained_state_dict(model, state_dict, is_finetune):
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            if name.startswith("module."):
                new_name = name.split("module.")[-1]
                if new_name.startswith("linear") and is_finetune:
                    print(name, " not loaded")
                    continue
                own_state[new_name].copy_(param)
            else:
                print(name, " not loaded")
                continue
        else:
            own_state[name].copy_(param)
    return model



def init_process_group(proc_rank, world_size, device_type="npu", port="29588", dist_backend="hccl"):
    """Initializes the default process group."""

    # Initialize the process group
    print("==================================")    
    print('Begin init_process_group')
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = port
    if device_type == "npu":
        torch.distributed.init_process_group(
            backend=dist_backend,
            world_size=world_size,
            rank=proc_rank
        )
    elif device_type == "cuda":
        torch.distributed.init_process_group(
            backend=dist_backend,
            init_method="tcp://{}:{}".format("127.0.0.1", port),
            world_size=world_size,
            rank=proc_rank
        )        

    print("==================================")
    print("Done init_process_group")

    # Set the GPU to use
    if device_type == "npu":
        torch.npu.set_device(proc_rank)
    elif device_type == "cuda":
        torch.cuda.set_device(proc_rank)
    print('Done set device', device_type, dist_backend, world_size, proc_rank)


def profiling(loader, model, loss_fun, optimizer, loc, args):
    # switch to train mode
    model.train()

    def update(model, images, target, optimizer):
        output = model(images)
        loss = loss_fun(output, target)
        optimizer.zero_grad()
        if args.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()        
        optimizer.step()

    for i, (images, target) in enumerate(loader):
        if 'npu' in args.device:
            target = target.to(torch.int32)

        if 'npu' in args.device or 'cuda' in args.device:
            images = images.to(loc, non_blocking=True)
            target = target.to(loc, non_blocking=True)
            
        if i < 5:
            update(model, images, target, optimizer)
        else:
            if args.device == 'npu':
                with torch.autograd.profiler.profile(use_npu=True) as prof:
                    update(model, images, target, optimizer)
            elif args.device == "cuda":
                with torch.autograd.profiler.profile(use_cuda=True) as prof:
                    update(model, images, target, optimizer)
            break

    prof.export_chrome_trace("output_npu.prof")


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

    if args.num_gpus > 1:
        init_process_group(proc_rank=args.rank_id, world_size=args.num_gpus, device_type=args.device)
    elif args.device == "npu":
        torch.npu.set_device(args.rank_id)
    elif args.device == "cuda":
        torch.cuda.set_device(0)

    main_worker(args)


def main_worker(args):
    global best_acc1
    log_file = 'train_{}_{}_exp.txt'.format(args.arch, args.tag)


    args.batch_size = int(args.batch_size / args.num_gpus)
    args.workers = int((args.workers + args.num_gpus - 1) / args.num_gpus)

    loc = ""
    if args.device == "npu":
        cur_device = torch.npu.current_device()
        loc = "npu:" + str(cur_device)
    elif args.device == "cuda":
        cur_device = torch.cuda.current_device()
        loc = "cuda:" + str(cur_device)
    print("cur device: ", cur_device)
    
    repvgg_build_func = get_RepVGG_func_by_name(args.arch)
    if args.finetune:
        model = repvgg_build_func(num_classes=args.fclasses, deploy=False)
        checkpoint = torch.load(args.fresume, map_location=loc)
        load_my_pretrained_state_dict(model, checkpoint['state_dict'], args.finetune)     
    else:
        model = repvgg_build_func(deploy=False)
   
    model = model.npu()

    is_main = args.num_gpus == 1 or (args.num_gpus > 1 and args.rank_id == 0)

    if is_main:
        """
        for n, p in model.named_parameters():
            print(n, p.size())
        for n, p in model.named_buffers():
            print(n, p.size())
        """
        log_msg('epochs {}, lr {}, weight_decay {}'.format(args.epochs, args.lr, args.weight_decay), log_file)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().npu()

    optimizer = sgd_optimizer(model, args.lr, args.momentum, args.weight_decay, args.custom_weight_decay)

    lr_scheduler = CosineAnnealingLR(optimizer=optimizer,
                                     T_max=args.epochs * IMAGENET_TRAINSET_SIZE // args.batch_size // args.num_gpus)

    if args.amp:
        if hasattr(torch.npu.utils, 'is_support_inf_nan') and torch.npu.utils.is_support_inf_nan():
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level,
                                                loss_scale='dynamic', combine_grad=True)
        else:
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level,
                                                loss_scale=args.loss_scale_value, combine_grad=True)

    if args.num_gpus > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cur_device], broadcast_buffers=False)
            
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['scheduler'])
            if args.amp:
                amp.load_state_dict(checkpoint['amp'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    train_sampler, train_loader = get_default_ImageNet_train_sampler_loader(args)
    val_loader = get_default_ImageNet_val_loader(args)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    if args.profile:
        profiling(train_loader, model, criterion, optimizer, loc, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, lr_scheduler, is_main=is_main)

        if is_main:
            # evaluate on validation set
            acc1 = validate(val_loader, model, criterion, args)
            msg = '{}, epoch {}, acc {}'.format(args.arch, epoch, acc1)
            log_msg(msg, log_file)

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            
            if args.amp:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer' : optimizer.state_dict(),
                    'scheduler': lr_scheduler.state_dict(),
                    'amp' : amp.state_dict(),
                }, is_best, filename = '{}_{}.pth.tar'.format(args.arch, args.tag),
                    best_filename='{}_{}_best.pth.tar'.format(args.arch, args.tag))
            else:               
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer' : optimizer.state_dict(),
                    'scheduler': lr_scheduler.state_dict(),
                }, is_best, filename = '{}_{}.pth.tar'.format(args.arch, args.tag),
                    best_filename='{}_{}_best.pth.tar'.format(args.arch, args.tag))


def train(train_loader, model, criterion, optimizer, epoch, args, lr_scheduler, is_main):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5, ],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).view(1, 3, 1, 1)
    std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).view(1, 3, 1, 1)
    device = torch.device('npu')
    mean = mean.to(device, non_blocking=True)
    std = std.to(device, non_blocking=True)
    end = time.time()
    profile = Profile(start_step=int(os.getenv('PROFILE_START_STEP', 10)),
                      profile_type=os.getenv('PROFILE_TYPE'))

    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
            
        if 'npu' in args.device:
            target = target.to(torch.int32)
            
        if 'npu' in args.device:
            images = images.npu(non_blocking=True).permute(0, 3, 1, 2).to(torch.float).sub(mean).div(std)
            target = target.npu(non_blocking=True)
            
        profile.start()
        # compute output
        output = model(images)

        loss = criterion(output, target)
        if args.custom_weight_decay:
            for module in model.modules():
                if hasattr(module, 'get_custom_L2'):
                    loss += args.weight_decay * 0.5 * module.get_custom_L2()

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
        profile.end()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        lr_scheduler.step()

        if i == 4:
            batch_time.reset()
        if is_main and i % args.print_freq == 0:
            progress.display(i)
        if is_main and i % 1000 == 0:
            print('cur lr: ', lr_scheduler.get_lr()[0])
    
    if is_main:
       print("[npu id:", args.rank_id, "]", '* FPS@all {:.3f}'.format(args.num_gpus * args.batch_size / batch_time.avg))


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
            if 'npu' in args.device:
                target = target.to(torch.int32)

            if 'npu' in args.device:
                images = images.npu(non_blocking=True)
                target = target.npu(non_blocking=True)

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

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename, best_filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)


if __name__ == '__main__':
    main()
