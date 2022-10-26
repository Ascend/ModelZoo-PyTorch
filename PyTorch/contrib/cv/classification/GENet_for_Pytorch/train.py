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
# ============================================================================

import sys
import torch
if torch.__version__ >= '1.8':
    import torch_npu

import argparse
import json
import os
import shutil
import time
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.model_zoo as model_zoo
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms
import torch.multiprocessing as mp
import random
from torch import nn
from models import *
from models.wideresnet import WideResNet
from utils import *
import apex
from apex import amp

import warnings

#Basic
parser = argparse.ArgumentParser(description='Pruning')
parser.add_argument('-j', '--workers', default=9, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--device', default='npu', type=str, help='npu or gpu')
parser.add_argument('--eval', '-e', action='store_true', help='resume from checkpoint')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model for evaling')
parser.add_argument('--print_freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')

#Distributed training
parser.add_argument('--multiprocessing_distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--world_size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', type=int, default=-1, help='local rank.')
parser.add_argument('--dist_url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

#Learning specific arguments
parser.add_argument('-b', '--batch_size', default=128, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-bt', '--test_batch_size', default=128, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-lr', '--learning_rate', default=.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--lr_decay_ratio', default=0.2, type=float, help='learning rate decay factor')
parser.add_argument('-epochs', '--no_epochs', default=300, type=int, metavar='epochs', help='no. epochs')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--nesterov', default=False, type=bool, help='yesterov?')
parser.add_argument('--weight_decay', '--wd', default=0.0005, type=float, metavar='W', help='weight decay')
parser.add_argument('--epoch_step', default='[60,120,160,200]', type=str, help='json list with epochs to drop lr on')

#Dataset
parser.add_argument('--dataset', choices=['cifar10','cifar100'], default = 'cifar10')
parser.add_argument('--DataPath', default='/opt/npu/', type=str, help='where is dataset')

#Net specific
parser.add_argument('--depth', '-d', default=16, type=int, metavar='D', help='wrn depth')
parser.add_argument('--width', '-w', default=8, type=int, metavar='W', help='wrn width)')
parser.add_argument('--mlp', default=False, type=bool, help='mlp?')
parser.add_argument('--extra_params', default=False, type=bool, help='extraparams?')
parser.add_argument('--extent', default=0, type=int, help='Extent for pooling')

#For Ascend 910 NPU
parser.add_argument('--addr', default='192.168.88.163',
                    type=str, help='master addr')
parser.add_argument('--device_list', default='0,1,2,3,4,5,6,7',
                    type=str, help='device id list')
parser.add_argument('--amp', default=False, action='store_true',
                    help='use amp to train the model')
parser.add_argument('--loss_scale', default=128.0, type=float,
                    help='loss scale using in amp, default -1 means dynamic')
parser.add_argument('--opt_level', default='O2', type=str,
                    help='loss scale using in amp, default -1 means dynamic')
parser.add_argument('--prof', default=False, action='store_true',
                    help='use profiling to evaluate the performance of model')

args = parser.parse_args()

# Dataset and Preprocess
best_error1 = 100
error_history = []

if args.device == 'npu':
    import torch.npu

# funtion

def flush_print(func):
    def new_print(*args, **kwargs):
        func(*args, **kwargs)
        sys.stdout.flush()
    return new_print

def device_id_to_process_device_map(device_list):
    devices = device_list.split(",")
    devices = [int(x) for x in devices]
    devices.sort()

    process_device_map = dict()
    for process_id, device_id in enumerate(devices):
        process_device_map[process_id] = device_id
    return process_device_map

def save_checkpoint(state, is_best, filename='checkpoints/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'checkpoints/model_best.pth.tar')

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
    
def main_worker(gpu, ngpus_per_node, args):
    #Dataset and Preprocess
    epoch_step = json.loads(args.epoch_step)
    if not os.path.exists('checkpoints/'):
        os.makedirs('checkpoints/')

    # GET NORMS FOR DATASET
    if args.dataset == 'cifar10':
        MEAN = (0.4914, 0.4822, 0.4465)
        STD  = (0.2023, 0.1994, 0.2010)
        NO_CLASSES = 10

    elif args.dataset == 'cifar100':
        MEAN =  (0.5071, 0.4867, 0.4408)
        STD =    (0.2675, 0.2565, 0.2761)
        NO_CLASSES = 100
    else:
        raise ValueError('pick a dataset')
    print('Standard Aug')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN,STD),
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    if args.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=args.DataPath,
                                            train=True, download=True, transform=transform_train)
        valset = torchvision.datasets.CIFAR10(root=args.DataPath,
                                              train=False, download=True, transform=transform_val)
    elif args.dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root=args.DataPath,
                                                train=True, download=False, transform=transform_train)
        valset = torchvision.datasets.CIFAR100(root=args.DataPath,
                                              train=False, download=False, transform=transform_val)

    else:
        raise ValueError('Pick a dataset (ii)')
    global best_error1
    
    args.gpu = args.process_device_map[gpu]
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
            print(args.rank)
        if args.device == 'npu':
            dist.init_process_group(backend='hccl',  #init_method=args.dist_url,
                                    world_size=args.world_size, rank=args.rank)
            print('======>Using npu distributed!')
        else:
            dist.init_process_group(backend='nccl', init_method=args.dist_url,
                                    world_size=args.world_size, rank=args.rank)
            print('======>Using gpu distributed!')
                                    
    #model
    if args.pretrained:
        print("=> using pre-trained!")
        model = WideResNet(args.depth, args.width, num_classes=NO_CLASSES, mlp=args.mlp, extra_params=args.extra_params)
        print("loading best model!")
        pretrained_dict = torch.load("checkpoints/model_best.pth.tar", map_location="cpu")["state_dict"]
        model.load_state_dict({k.replace('module.',''):v for k, v in pretrained_dict.items()})
        if "fc.weight" in pretrained_dict:
            pretrained_dict.pop('fc.weight')
            pretrained_dict.pop('fc.bias')
        model.load_state_dict(pretrained_dict, strict=False)     
    else:
        print("=> creating model!")
        model = WideResNet(args.depth, args.width, num_classes=NO_CLASSES, mlp=args.mlp, extra_params=args.extra_params)
    
    #get_no_params(model)
        
    if args.distributed:
        if args.gpu is not None:
            if args.device == 'npu':
                loc = 'npu:{}'.format(args.gpu)
                torch.npu.set_device(loc)
                model = model.to(loc)
            else:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / args.world_size)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        else:
            if args.device == 'npu':
                loc = 'npu:{}'.format(args.gpu)
                model = model.to(loc)
            else:
                model.cuda()
            print("[device id:", args.gpu, "]")
                  
    elif args.gpu is not None:
        print("[device id:", args.gpu, "]")
        if args.device == 'npu':
            loc = 'npu:{}'.format(args.gpu)
            torch.npu.set_device(args.gpu)
            model = model.to(loc)
        else:
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        print("[device id:", args.gpu, "]")
        if args.device == 'npu':
            loc = 'npu:{}'.format(args.gpu)
        else:
            print("before : model = torch.nn.DataParallel(model).cuda()")
    
    #optimizer 
    if args.amp:       
        optimizer = apex.optimizers.NpuFusedSGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
        
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=epoch_step, gamma=args.lr_decay_ratio, last_epoch=-1)   
    #amp 
    if args.amp:
        print('Using amp!')
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level, loss_scale=args.loss_scale, combine_grad=True)
    
    #DDP  
    if args.distributed:
        if args.gpu is not None:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], broadcast_buffers=False)
        else:
            print("[device id:", args.gpu, "]")
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        print("[device id:", args.gpu, "]")
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        print("[device id:", args.gpu, "]")
        if args.device == 'npu':
            loc = 'npu:{}'.format(args.gpu)
            model = torch.nn.DataParallel(model).to(loc)
        else:
            model = torch.nn.DataParallel(model).cuda()  
    
    # loss 
    if args.device == 'npu':
        loc = 'npu:{}'.format(args.gpu)
        criterion = nn.CrossEntropyLoss().to(loc)
    else:
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)  
    
    #resume    
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
            best_error1 = checkpoint['best_err1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if args.amp:
                amp.load_state_dict(checkpoint['amp'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        args.start_epoch = 0    
    cudnn.benchmark = True
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                          shuffle=(train_sampler is None),
                                          num_workers=args.workers,
                                          pin_memory=True,
                                          sampler=train_sampler if not args.eval else None,
                                          drop_last=True)
    else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                          num_workers=args.workers,
                                          pin_memory=False,
                                          shuffle=True)

    valloader = torch.utils.data.DataLoader(valset, batch_size=args.test_batch_size, shuffle=False if not args.distributed else True,
                                        num_workers=args.workers,
                                        pin_memory=False)
                                        
    if args.eval:
        validate(valloader, model, criterion, args, ngpus_per_node)
        return
        
    if args.prof:
        profiling(trainloader, model, criterion, optimizer, args)
        return

    #train   
    start_time = time.time()
    for epoch in range(args.start_epoch, args.no_epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)        
        print('Epoch %d:' % epoch)
        print('Learning rate is %s' % [v['lr'] for v in optimizer.param_groups][0])
        train(trainloader, model, criterion, optimizer, epoch, args, ngpus_per_node)
        scheduler.step()
        err1 = validate(valloader, model, criterion, args, ngpus_per_node)
        is_best = err1 < best_error1
        best_error1 = min(err1, best_error1)
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            if args.amp:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': 'GENET',
                    'state_dict': model.state_dict(),
                    'best_err1': best_error1,
                    'optimizer': optimizer.state_dict(),
                    'amp': amp.state_dict(),
                }, is_best)
            else:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': 'GENET',
                    'state_dict': model.state_dict(),
                    'best_err1': best_error1,
                    'optimizer': optimizer.state_dict(),
                }, is_best)
    
def main():
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True                         

    os.environ['MASTER_ADDR'] = args.addr
    os.environ['MASTER_PORT'] = '29688'

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    args.process_device_map = device_id_to_process_device_map(args.device_list)
    print(args.process_device_map)
    if args.device == 'npu':
        ngpus_per_node = len(args.process_device_map)
    else:
        if args.distributed:
            ngpus_per_node = torch.cuda.device_count()
        else:
            ngpus_per_node = 1
    print('ngpus_per_node:', ngpus_per_node)

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)

def train(trainloader, model, criterion, optimizer, epoch, args, ngpus_per_node):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    progress = ProgressMeter(len(trainloader),[batch_time, data_time, losses, top1, top5], prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()

    for i, (input, target) in enumerate(trainloader):

        # measure data loading time
        data_time.update(time.time() - end)

        if args.device == 'npu':
            loc = 'npu:{}'.format(args.gpu)
            input = input.to(loc, non_blocking=True).to(torch.float)
            target = target.to(torch.int32).to(loc, non_blocking=True)
        else:
            input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
                           
        output = model(input)
        loss = criterion(output, target)

            # measure accuracy and record loss
        err1, err5 = get_error(output.detach(), target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))
            
        if args.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                optimizer.zero_grad()
                scaled_loss.backward()
        else:
            optimizer.zero_grad()
            loss.backward()
        optimizer.step()
        if args.device == 'npu':
            torch.npu.synchronize()

        # measure elapsed time
        if i == 9:
            batch_time.reset()
            data_time.reset()
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            if not args.multiprocessing_distributed or (args.rank % ngpus_per_node == 0) :
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Err@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Err@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(trainloader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))
                
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                and args.rank % ngpus_per_node == 0):
        if batch_time.avg:
            print("[npu id:", args.gpu, "]", "batch_size:", args.world_size * args.batch_size,
                  'Time: {:.3f}'.format(batch_time.avg), '* FPS@all {:.3f}'.format(
                    args.batch_size * args.world_size / batch_time.avg))      

def validate(valloader, model, criterion, args, ngpus_per_node):
    global error_history

    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    progress = ProgressMeter(len(valloader), [batch_time, losses, top1, top5], prefix='Test: ')    

    # switch to evaluate mode
    model.eval()

    end = time.time()

    for i, (input, target) in enumerate(valloader):

        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpu is not None:
            if args.device == 'npu':
                loc = 'npu:{}'.format(args.gpu)
                input = input.to(loc).to(torch.float)
            else:
                input = input.cuda(args.gpu, non_blocking=True)
            if args.device == 'npu':
                loc = 'npu:{}'.format(args.gpu)
                target = target.to(torch.int32).to(loc, non_blocking=True)
            else:
                loc = 'cuda:{}'.format(args.gpu)
                target = target.cuda(args.gpu, non_blocking=True)
        input, target = input.to(loc, non_blocking=True), target.to(loc, non_blocking=True)

        # compute output
        output = model(input)

        loss = criterion(output, target)

        # measure accuracy and record loss
        err1, err5 = get_error(output.detach(), target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # measure elapsed time
        if i == 9:
            batch_time.reset()
            data_time.reset()
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if not args.multiprocessing_distributed or (args.rank % ngpus_per_node == 0) :
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Err@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Err@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(valloader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
    if i % args.print_freq == 0:
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            print("[gpu id:", args.gpu, "]", '[AVG-ERR] * Err@1 {top1.avg:.3f} Err@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    elif (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0): 
        print(' * Err@1 {top1.avg:.3f} Err@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    # Record Top 1 for CIFAR
    error_history.append(top1.avg)
    return top1.avg

def profiling(data_loader, model, criterion, optimizer, args):
    # switch to train mode
    model.train()

    def update(model, input, target, optimizer):
        output = model(input)
        loss = criterion(output, target)
        if args.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.zero_grad()
        optimizer.step()

    for step, (input, target) in enumerate(data_loader):
        if args.device == 'npu':
            loc = 'npu:{}'.format(args.gpu)
            input = input.to(loc, non_blocking=True).to(torch.float)
            target = target.to(torch.int32).to(loc, non_blocking=True)
        else:
            input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            
        if step < 5:
            update(model, input, target, optimizer)
        else:
            if args.device == 'npu':
                with torch.autograd.profiler.profile(use_npu=True) as prof:
                    update(model, input, target, optimizer)
            else:
                with torch.autograd.profiler.profile(use_cuda=True) as prof:
                    update(model, input, target, optimizer)
            break
    #print(prof.key_averages().table(sort_by="self_cpu_time_total"))
    prof.export_chrome_trace("output.prof")

if __name__ == '__main__':
    main()
