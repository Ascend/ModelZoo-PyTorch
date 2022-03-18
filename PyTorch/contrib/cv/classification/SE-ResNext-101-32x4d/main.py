# Copyright 2021 Huawei Technologies Co., Ltd
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
# ============================================================================

import os
import shutil
import argparse
import sys
import time
import random
import warnings
import torch
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.distributed as dist
import torchvision
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
from apex import amp
import apex
import models

best_acc1 = 0


def main():
    parser = argparse.ArgumentParser(description='PyTorch Se-ResNeXt101 ImageNet Training')

    # dataset setting
    parser.add_argument('--data_path', metavar='DIR', default='/opt/npu/imagenet',
                        help='path to dataset')
    parser.add_argument('--workers', default=192, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    # training setting
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run') 
    parser.add_argument('--batch-size', default=128, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', default=0.6, type=float,
                        metavar='LR', help='initial learning rate', dest='lr') 
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    
    # apex setting
    parser.add_argument('--amp', default=True, action='store_true',
                        help='use amp to train the model')
    parser.add_argument('--opt-level', default="O2", type=str, help='apex optimize level')
    parser.add_argument('--loss-scale-value', default=None, type=float, help='static loss scale value')
    parser.add_argument('--combine-grad', default=True, action='store_true',
                        help='use amp to train the model')

    # basic distribution setting 
    parser.add_argument('--ddp', 
                        dest='ddp',
                        action='store_true',
                        help='use distribution training')
    parser.add_argument('--nodes', default=1, type=int,
                        help='number of data loading workers (default: 4)') 
    parser.add_argument('--node_rank', default=0, type=int,
                        help='ranking within the nodes') 
    parser.add_argument('--device_list', default='0,1,2,3,4,5,6,7', type=str, help='device id list')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    
    # other setting 
    parser.add_argument('--evaluate',
                        dest='evaluate',
                        action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--prof', 
                        dest='prof',
                        action='store_true',
                        help='print model profile on training')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--stop-step-num', default=None, type=int, 
                        help='after the stop-step, killing the training task')

    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--num_classes', default=1000, type=int,
                        help='The number of classes.')
    args = parser.parse_args() 
    
    print(args)
    
    args.distributed = (args.nodes > 1) or args.ddp 
    
    args.process_device_map = device_id_to_process_device_map(args.device_list)
    if args.distributed:
        ngpus_per_node = len(args.process_device_map)
    else:
        ngpus_per_node = 1

    print("Use multiprocessing for training :", args.distributed)

    args.world_size = ngpus_per_node * args.nodes
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12345'

    if args.seed is not None:
        seed_everything(args.seed)
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.distributed: 
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(0, ngpus_per_node, args)


def main_worker(process_id, ngpus_per_node, args): 
    # main worker process for every backend
    global best_acc1

    # each process npu/gpu device id
    deviceid = args.process_device_map[process_id]
    loc = "npu: {}".format(deviceid)
    if deviceid is not None:
        print("Use NPU: {} for training".format(deviceid))

    args.rank = args.node_rank * ngpus_per_node + process_id
     
    # mainporcess maintance print to log and torch.save
    args.mainprocess = (args.distributed is False) or (args.rank == 0)

    # thread nums of each loader
    args.batch_size = int(args.batch_size / ngpus_per_node)
    args.loader_workers = int(args.workers / ngpus_per_node)
    
    if args.distributed:
        # init hccl backend by setting
        dist.init_process_group(backend = 'hccl', 
                                #init_method='env://', 
                                world_size = args.world_size, 
                                rank = args.rank)
    
    torch.npu.set_device(loc) 
 
    model = models.seresnext101_32x4d(num_classes=args.num_classes)
    model = model.npu() 

    # optionally resume from a checkpoint
    if os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location=loc) 
        args.start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']
        state_dict = checkpoint['state_dict']

        if args.pretrained:
            # target model prediction weight
            if "fc.weight" in state_dict:
                state_dict.pop('fc.weight')
                state_dict.pop('fc.bias')
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(state_dict)
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
    else:
        args.start_epoch = 0 
        print("=> no checkpoint found at '{}'".format(args.resume))
    
    optimizer = apex.optimizers.NpuFusedSGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().npu()

    # Wrap the model 
    model, optimizer = amp.initialize(model, 
                                        optimizer, 
                                        opt_level = args.opt_level, 
                                        combine_grad = args.combine_grad, 
                                        loss_scale = args.loss_scale_value)
    if args.distributed: 
        if args.pretrained:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids = [deviceid], broadcast_buffers = False,
                                                                find_unused_parameters=True)
        else:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids = [deviceid], broadcast_buffers = False)

    cudnn.benchmark = True 

    # Data loading code 
    train_loader, val_loader, train_sampler = dataloader(args) 
 
    if args.evaluate:
        acc1 = validate(val_loader, model, criterion, deviceid, args) 
        return

    if args.prof:
        profiling(train_loader, model, criterion, optimizer, deviceid, args)
        return 

    for epoch in range(args.start_epoch, args.epochs):

        # sampler need set epoch 
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # adjust learning rate
        adjust_learning_rate(optimizer, epoch, args)  
        if args.mainprocess:
            print("lr adjust to :", optimizer.param_groups[0]['lr']) 
        train(train_loader, model, criterion, optimizer, epoch, deviceid, args)

        # exit process if just want to test the performance
        if args.stop_step_num is not None:  
            break

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, deviceid, args) 

        if args.mainprocess:
            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
 
            file_name = "checkpoint"

            if args.distributed:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            
            save_checkpoint(epoch, {
                'epoch': epoch + 1,
                'arch': "seresnext101",
                'state_dict': state_dict,
                'best_acc1': best_acc1, 
            }, is_best, file_name)


def profiling(train_loader, model, criterion, optimizer, deviceid, args, epoch=0):
    # switch to profiling mode
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    optimizer.zero_grad()
    end = time.time()
    images, target = next(iter(train_loader))    
    images = images.npu(non_blocking=True)
    target = target.long().npu(non_blocking=True)
 
    def onestep(images, target, model, criterion, optimizer): 
        # 
        output = model(images)
        loss = criterion(output, target.long())

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return output, target, loss
        
    for i in range(100): 
        # measure data loading time 
        if args.mainprocess:
            data_time.update(time.time() - end)   
        output, target, loss = onestep(images, target, model, criterion, optimizer)

        if args.mainprocess:   
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time() 
            progress.display(i)

    if args.mainprocess:
        print("start log prof") 
 
    with torch.autograd.profiler.profile(use_npu=True) as prof:
        output, target, loss = onestep(images, target, model, criterion, optimizer) 
    print(prof.key_averages().table())
    prof.export_chrome_trace("output.prof")


def train(train_loader, model, criterion, optimizer, epoch, deviceid, args):
    # switch to train mode
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
        images = images.npu(non_blocking=True)
        target = target.long().npu(non_blocking=True)
        if args.mainprocess:
            # measure data loading timetime
            data_time.update(time.time() - end)   
        
        output = model(images)
        loss = criterion(output, target.long())

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if args.mainprocess:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        if i % args.print_freq == 0 and args.mainprocess:
            progress.display(i)

        
        if args.stop_step_num is not None and i >= args.stop_step_num:
            break

    if args.mainprocess and batch_time.avg:
        print("[npu id:", deviceid, "]", "batch_size:", args.world_size * args.batch_size,
                'Time: {:.3f}'.format(batch_time.avg), '* FPS@all {:.3f}'.format(
                args.batch_size * args.world_size / batch_time.avg)) 


def validate(val_loader, model, criterion, deviceid, args): 
    # switch to validate mode
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
            images = images.npu(non_blocking=True) 
            target = target.to(torch.int32).npu(non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target.long())

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            
            if args.mainprocess:
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 and args.mainprocess:
                progress.display(i)

        if args.mainprocess: 
            print("[gpu id:", deviceid, "]", '[AVG-ACC] * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                    .format(top1=top1, top5=top5)) 
    return top1.avg


def save_checkpoint(epoch, state, is_best, filename='checkpoint'):
    # saving the checkpoint  
    filename2 = filename + ".pth"
    torch.save(state, filename2)
    if is_best:
        shutil.copyfile(filename2, filename + 'model_best.pth')


def dataloader(args):
    """ Create training & validation dataloader """ 
    traindir = os.path.join(args.data_path, 'train')
    valdir = os.path.join(args.data_path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    # Create Dataset
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    
    # Create Sampler
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                            num_replicas=args.world_size,
                                            rank=args.rank)
 
    else:
        train_sampler = None 

    # Create Loader
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=(train_sampler is None),
                                                num_workers=args.loader_workers,
                                                drop_last=True,
                                                pin_memory=False,
                                                sampler=train_sampler)
    
    val_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                                batch_size=args.batch_size, 
                                                shuffle=False, 
                                                drop_last=True,
                                                num_workers=args.loader_workers, 
                                                pin_memory=False,
                                                sampler=None)
 
    return train_loader, val_loader, train_sampler


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
            temp = correct[:k]
            correct_k = temp.reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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
    """process the data saved by AverageMeter"""
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


def flush_print(func):
    """for servers to immediately record the logs"""
    def new_print(*args, **kwargs):
        """set a new system standary outpu flush"""
        func(*args, **kwargs)
        sys.stdout.flush()
    return new_print


def device_id_to_process_device_map(device_list):
    # devied string like "3,4" to list
    devices = device_list.split(",")
    devices = [int(x) for x in devices]
    devices.sort()

    process_device_map = dict()
    for process_id, device_id in enumerate(devices):
        process_device_map[process_id] = device_id

    return process_device_map


def seed_everything(seed):
    # seeding
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True


if __name__ == '__main__':
    # print = flush_print(print)
    main()