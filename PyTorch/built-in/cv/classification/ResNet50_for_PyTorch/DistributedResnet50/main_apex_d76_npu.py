# -*- coding: utf-8 -*-
# BSD 3-Clause License
#
# Copyright (c) 2017,
# All rights reserved.
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

from apex import amp

#########################################################################
#NV 代码移植
#########################################################################
import image_classification.resnet as nvmodels
from image_classification.multi_epochs_dataloader import MultiEpochsDataLoader 
from image_classification.smoothing import LabelSmoothingGpu
from image_classification.smoothing import CrossEntropy
from image_classification.mixup import NLLMultiLabelSmooth, MixUpWrapper

import image_classification.logger as log
from image_classification.training import *
#########################################################################

BATCH_SIZE = 512
OPTIMIZER_BATCH_SIZE=2048

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data',
                    metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch',
                    metavar='ARCH',
                    default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers',
                    default=32,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--num_classes',
                    default=1000,
                    type=int,
                    metavar='N',
                    help='class number of dataset')
parser.add_argument('--save_ckpt_path',
                    metavar='DIR',
                    default='./',
                    help='path of checkpoint file')
parser.add_argument('--epochs',
                    default=90,
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
                    default=2.048,
                    type=float,
                    metavar='LR',
                    help='initial learning rate',
                    dest='lr')
parser.add_argument('--momentum',
                    default=0.875,
                    type=float,
                    metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay',
                    default=3.0517578125e-05,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 1e-4)',
                    dest='weight_decay')

model_configs = nvmodels.resnet_configs.keys()
parser.add_argument('--model-config', '-c',
                    metavar='CONF',
                    default='classic',
                    choices=model_configs,
                    help='model configs: ' +
                         ' | '.join(model_configs) +
                         '(default: classic)')
parser.add_argument('--lr-schedule',
                    default='cosine',
                    type=str,
                    metavar='SCHEDULE',
                    choices=['step', 'linear', 'cosine'],
                    help='Type of LR schedule: {}, {}, {}'.format(
                         'step', 'linear', 'cosine'))
parser.add_argument('--warmup',
                    default=8,
                    type=int,
                    metavar='E',
                    help='number of warmup epochs')
parser.add_argument('--label-smoothing',
                    default=0.0,
                    type=float,
                    metavar='S',
                    help='label smoothing')
parser.add_argument('--mixup',
                    default=0.0,
                    type=float,
                    metavar='ALPHA',
                    help='mixup alpha')
parser.add_argument('--bn-weight-decay',
                    action='store_true',
                    help='use weight_decay on batch normalization learnable parameters, (default: false)')
parser.add_argument('--static-loss-scale',
                    type=float,
                    default=128,
                    help='Static loss scale, positive power of 2 values can improve fp16 convergence.')
parser.add_argument('--dynamic-loss-scale',
                    action='store_true',
                    help='Use dynamic loss scaling.  If supplied, this argument supersedes ' +
                         '--static-loss-scale.')
parser.add_argument('--nesterov',
                    action='store_true',
                    help='use nesterov momentum, (default: false)')
parser.add_argument('--amp',
                    action='store_true',
                    help='Run model AMP (automatic mixed precision) mode.')
parser.add_argument('--fp16',
                    action='store_true',
                    help='Run model fp16 mode.')
parser.add_argument('--workspace',
                    type=str,
                    default='./',
                    metavar='DIR',
                    help='path to directory where checkpoints will be stored')
parser.add_argument('--raport-file',
                    default='experiment_raport.json',
                    type=str,
                    help='file in which to store JSON experiment raport')
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
                    action='store_true',
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
parser.add_argument('--multiprocessing-distributed',
                    action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('-bm', '--benchmark',
                    default=0,
                    type=int,
                    metavar='N',
                    help='set benchmark status (default: 1,run benchmark)')
parser.add_argument('--device',
                    default='npu',
                    type=str,
                    help='npu or gpu')
parser.add_argument('--device-list',
                    default='0,1,2,3,4,5,6,7',
                    type=str,
                    help='device id list')
parser.add_argument('--addr',
                    default='10.136.181.115',
                    type=str,
                    help='master addr')
parser.add_argument('--checkpoint-nameprefix',
                    default='checkpoint',
                    type=str,
                    help='checkpoint-nameprefix')
parser.add_argument('--checkpoint-freq',
                    default=10,
                    type=int,
                    metavar='N',
                    help='checkpoint frequency (default: 0)'
                         '0: save only one file whitch per epoch;'
                         'n: save diff file per n epoch'
                         '-1:no checkpoint,not support')
parser.add_argument('-t',
                    '--fine-tuning',
                    action='store_true',
                    help='transfer learning + fine tuning - train only the last FC layer.')
# 图模式
parser.add_argument('--graph_mode',
                    action='store_true',
                    help='whether to enable graph mode.')

# 二进制
parser.add_argument('--bin_mode',
                    action='store_true',
                    help='whether to enable binary mode.')

# 精度模式
parser.add_argument('--precision_mode',
                    default='allow_mix_precision',
                    type=str,
                    help='precision_mode')

best_acc1 = 0

def nvidia_model_config(args):
    model = ResNet(builder,              #    builder,
                    Bottleneck,           #    version['block'],
                    4,                    #    version['expansion'],
                    [3, 4, 6, 3],         #    version['layers'],
                    [64, 128, 256, 512],  #    version['widths'],
                    1000)                 #    version['num_classes'])
    return model

def nvidia_logger_init(args):
    if False:
        logger = log.Logger(args.print_freq, [
            dllogger.StdOutBackend(dllogger.Verbosity.DEFAULT,
                               step_format=log.format_step),
            dllogger.JSONStreamBackend(
                dllogger.Verbosity.VERBOSE,
                os.path.join(args.workspace, args.raport_file))
        ])
    else:
        logger = log.Logger(args.print_freq, [])
    logger.log_parameter(args.__dict__, verbosity=dllogger.Verbosity.DEFAULT)
    args.logger = logger

#--label-smoothing 0.1 命令参数中用到，需调测
def nvidia_mixup_and_label_smoothing_getlossfunction(args):
    loss = nn.CrossEntropyLoss
    if args.mixup > 0.0: #mixup命令参数中未用到，暂不调测
        loss = lambda: NLLMultiLabelSmooth(args.label_smoothing)
    elif args.label_smoothing > 0.0: #--label-smoothing 0.1 命令参数中用到，需调测
        if args.device == 'npu':
           loss = lambda: CrossEntropy(args.label_smoothing)
        if args.device == 'gpu':
           loss = lambda: LabelSmoothingGpu(args.label_smoothing)

    if args.gpu is not None:
        if args.device == 'npu':
            loc = 'npu:{}'.format(args.gpu)
            criterion = loss().to(loc)
        else:
            criterion = loss().cuda(args.gpu) 
    return criterion

def nvidia_mixup_get_train_loader_iter(args):
    if args.mixup != 0.0: #mixup命令参数中未用到，暂不调测
        train_loader = MixUpWrapper(args.mixup, 1000, train_loader)

def nvidia_lr_policy(args):
    logger=args.logger
    if args.lr_schedule == 'step':
        lr_policy = lr_step_policy(args.lr, [30, 60, 80],
                                0.1,
                                args.warmup,
                                logger=logger)
    elif args.lr_schedule == 'cosine':
        lr_policy = lr_cosine_policy(args.lr,
                                    args.warmup,
                                    args.epochs,
                                    logger=logger)
    elif args.lr_schedule == 'linear':
        lr_policy = lr_linear_policy(args.lr,
                                    args.warmup,
                                    args.epochs,
                                    logger=logger)
    return lr_policy

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    print(args)

    if args.precision_mode == "must_keep_origin_dtype":
        args.fp16 = False
        option = {}
        option["ACL_PRECISION_MODE"] = "must_keep_origin_dtype" 
        torch.npu.set_option(option)
        torch.npu.config.allow_internal_format=False

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    os.environ['MASTER_ADDR'] = args.addr  # '10.136.181.51'
    os.environ['MASTER_PORT'] = '29501'

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
        ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        # The child process uses the environment variables of the parent process,
        # we have to set KERNEL_NAME_ID for every proc
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    if args.bin_mode:
        print('use binary mode')
        torch.npu.set_compile_mode(jit_compile=False)
    global best_acc1
    args.gpu = args.process_device_map[gpu]

    if args.gpu is not None:
        print("[npu id:",args.gpu,"]","Use NPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu

        if args.device == 'npu':
            args.rank = int(os.getenv("NODE_RANK", 0)) * ngpus_per_node + args.rank
            print("the global_rank is :", args.rank)
            dist.init_process_group(backend=args.dist_backend,
                                    world_size=args.world_size, rank=args.rank)
        else:
            dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                    world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        print("[gpu id:",args.gpu,"]","=> using pre-trained model '{}'".format(args.arch))
        model = nvmodels.build_resnet("resnet50", "classic", True)
        pretrained_dict = \
        torch.load("/home/checkpoint_npu0model_best.pth.tar", map_location="cpu")["state_dict"]
        pretrained_dict.pop('module.fc.weight')
        pretrained_dict.pop('module.fc.bias')
        model.load_state_dict(pretrained_dict, strict=False)
    else:
        print("[npu id:",args.gpu,"]","=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](num_classes=args.num_classes)

    if args.fine_tuning:
        print("=> transfer learning + fine tuning(train only the last FC layer)")
        for param in model.parameters():
            param.requires_grad = True
        if args.arch == "resnet50":
            model.parameters()
        else:
            print("Error: Fine-tuning is not supported on this architecture")
            exit(-1)
    else:
        model.parameters()

    nvidia_logger_init(args)
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
    elif args.gpu is not None:
        if args.device == 'npu':
            loc = 'npu:{}'.format(args.gpu)
            torch.npu.set_device(args.gpu)
            model = model.to(loc)
        else:
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)

    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            print("[gpu id:",args.gpu,"]","============================test   2==========================")
        else:
            if args.device == 'npu':
                loc = 'npu:{}'.format(args.gpu)
            else:
                print("before : model = torch.nn.DataParallel(model).cuda()")
            
    optimizer_state = None
    optimizer = get_optimizer(list(model.named_parameters()),
                            args.fp16,
                            args.lr,
                            args.momentum,
                            args.weight_decay,
                            nesterov=args.nesterov,
                            bn_weight_decay=args.bn_weight_decay,
                            state=optimizer_state,
                            static_loss_scale=args.static_loss_scale,
                            dynamic_loss_scale=args.dynamic_loss_scale)

    lr_scheduler = nvidia_lr_policy(args)
    if args.precision_mode == "must_keep_origin_dtype":
        model, optimizer = amp.initialize(model, optimizer, opt_level="O0",verbosity=1)
    else:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O2",loss_scale = 1024,verbosity=1)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            # args.batch_size = int(args.batch_size / ngpus_per_node)
            # args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            if args.pretrained:
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                                  broadcast_buffers=False, find_unused_parameters=True)
            else:
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], broadcast_buffers=False)
        else:
            model = torch.nn.parallel.DistributedDataParallel(model)

    elif args.gpu is not None:
        print("[gpu id:",args.gpu,"]","=======================test   elif args.gpu is not None:======================")

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

    # define loss function (criterion) and optimizer
    if args.device == 'npu':
        loc = 'npu:{}'.format(args.gpu)
        criterion = nn.CrossEntropyLoss().to(loc)
    else:
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    criterion = nvidia_mixup_and_label_smoothing_getlossfunction(args)#需增加设备类型参数(npu/gpu)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("[npu id:",args.gpu,"]","=> loading checkpoint '{}'".format(args.resume))
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
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("[npu id:",args.gpu,"]","=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("[npu id:",args.gpu,"]","=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    train_loader, train_loader_len, sampler = get_pytorch_train_loader(args.data, args.batch_size,
                                                                       workers=args.workers, distributed=args.distributed)
    
    val_loader = get_pytorch_val_loader(args.data, args.batch_size, args.workers, distributed=False)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return
    
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler.set_epoch(epoch)

        # train for one epoch
        acc1 = train(train_loader, train_loader_len, model, criterion, optimizer, epoch, args, ngpus_per_node, lr_scheduler)

        # evaluate on validation set
        if (epoch + 1) % 1 == 0:
            if args.device == 'npu' and args.gpu != 0:
                continue
            acc1 = validate(val_loader, model, criterion, args,ngpus_per_node)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        if args.device == 'npu' and args.gpu == 0 and epoch == 89:
            print("Complete 90 epoch training, take time:{}h".format(round((time.time() - start_time) / 3600.0, 2)))
        
        if args.device == 'gpu':
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                    and args.rank % ngpus_per_node == 0):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, args)
        elif args.device == 'npu':
            #保存恢复点
            if (not args.multiprocessing_distributed or (args.multiprocessing_distributed
                   and args.rank % ngpus_per_node == 0 and epoch == args.epochs - 1) ):
                #单P情况，每个epoch均保存checkpoint文件
                #多P情况，仅最后一个epoch，保存rank 0的checkpoint文件 
                filename = args.checkpoint_nameprefix + ".pth.tar"

                modeltmp = model.cpu()
                #保留最后一个epoch的checkpoint，防止异常退出
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': modeltmp.state_dict(),
                    'best_acc1': best_acc1,
                }, is_best, args, filename=filename)
                
                if (epoch == (args.epochs - 1)) or ((args.checkpoint_freq > 0) and (((epoch+1) % args.checkpoint_freq) == 0)):
                    #保留每个freq的checkpoint，共epochs/freq个checkpoint文件
                    #最后一个epoch保存独立的checkpoint文件
                    #每隔freq个epoch保存一个checkpoint文件
                    filename=args.checkpoint_nameprefix + "-epoch"+str(epoch) + ".pth.tar"
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': modeltmp.state_dict(),
                        'best_acc1': best_acc1,
                    }, is_best, args, filename=filename)

                loc = 'npu:{}'.format(args.gpu)
                modeltmp.to(loc)

def train(train_loader, train_loader_len, model, criterion, optimizer, epoch, args,ngpus_per_node, lr_scheduler):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        train_loader_len,
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).view(1, 3, 1, 1)
    std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).view(1, 3, 1, 1)
    if args.device == 'npu' and args.gpu is not None:
        loc = 'npu:{}'.format(args.gpu)
        mean = mean.to(loc, non_blocking=True)
        std = std.to(loc, non_blocking=True)

    end = time.time()
    
    if args.benchmark == 1 :
        optimizer.zero_grad()
    for i, (images, target) in enumerate(train_loader):
        if i > 99:pass
        # 图模式
        if args.graph_mode:
            print("graph mode on")
            torch.npu.enable_graph_mode()
        # measure data loading time
        data_time.update(time.time() - end)

        lr_scheduler(optimizer, i, epoch)

        if args.device == 'npu':
            loc = 'npu:{}'.format(args.gpu)
            # 图模式
            if args.graph_mode:
                images = images.to(loc, non_blocking=True)
                target = target.to(loc, non_blocking=True)
                images = images.to(torch.float).sub(mean).div(std)
                target = target.to(torch.int32)
            else:
                images = images.to(loc, non_blocking=True).to(torch.float).sub(mean).div(std)
                target = target.to(torch.int32).to(loc, non_blocking=True)
        else:
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)

        loss = criterion(output, target)

        # measure accuracy and record loss
        # 图模式
        if not args.graph_mode:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        if args.benchmark == 0 :
            optimizer.zero_grad()

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        if args.benchmark == 0 :
            optimizer.step()
        elif args.benchmark == 1 :
            BATCH_SIZE_multiplier = int(OPTIMIZER_BATCH_SIZE / args.batch_size)
            BM_optimizer_step = ((i + 1) % BATCH_SIZE_multiplier) == 0
            if BM_optimizer_step:
                for param_group in optimizer.param_groups:
                    for param in param_group['params']:
                        param.grad /= BATCH_SIZE_multiplier
                optimizer.step()
                optimizer.zero_grad()
        # 图模式        
        if not args.graph_mode:
            torch.npu.synchronize()

        # 图模式
        if args.graph_mode:
            print("graph mode launch")
            torch.npu.launch_graph()
            if i == len(train_loader):
                torch.npu.synchronize()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                    and args.rank % ngpus_per_node == 0):
                    progress.display(i)
    # 图模式
    if args.graph_mode:
        print("graph mode off")
        torch.npu.disable_graph_mode()   
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
            and args.rank % ngpus_per_node == 0):
        if batch_time.avg > 0:
            print("[npu id:",args.gpu,"]", "batch_size:", ngpus_per_node*args.batch_size, 'Time: {:.3f}'.format(batch_time.avg), '* FPS@all {:.3f}'.format(
                    ngpus_per_node*args.batch_size/batch_time.avg))
    return top1.avg

def validate(val_loader, model, criterion, args,ngpus_per_node):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).view(1, 3, 1, 1)
    std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).view(1, 3, 1, 1)
    if args.device == 'npu' and args.gpu is not None:
        loc = 'npu:{}'.format(args.gpu)
        mean = mean.to(loc, non_blocking=True)
        std = std.to(loc, non_blocking=True)

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if i > 50:pass
            if args.gpu is not None:
                if args.device == 'npu':
                    loc = 'npu:{}'.format(args.gpu)
                    images = images.to(loc).to(torch.float).sub(mean).div(std)
                else:
                    images = images.cuda(args.gpu, non_blocking=True)
            if args.device == 'npu':
                loc = 'npu:{}'.format(args.gpu)
                target = target.to(torch.int32).to(loc, non_blocking=True)
            else:
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)
            target = target.to(torch.int32).to(loc, non_blocking=True)
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

        if i % args.print_freq == 0:
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                    and args.rank % ngpus_per_node == 0):
                print("[npu id:",args.gpu,"]",'[AVG-ACC] * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                        .format(top1=top1, top5=top5))
    return top1.avg

def save_checkpoint(state, is_best, args, filename='checkpoint.pth.tar'):
    filename2 = os.path.join(args.save_ckpt_path, filename)
    torch.save(state, filename2)
    if is_best:
        shutil.copyfile(filename2, os.path.join(args.save_ckpt_path, 'model_best.pth.tar'))

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
            self.batch_size = n
        
        self.val = val
        self.count += n
        if self.count > (self.start_count_index * self.batch_size):
            self.sum += val * n
            self.avg = self.sum / (self.count - self.start_count_index * self.batch_size)

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
        print("[npu id:","0","]",'\t'.join(entries))

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

def get_pytorch_train_loader(data_path, batch_size, workers=5, _worker_init_fn=None, distributed=False):
    traindir = os.path.join(data_path, 'train')
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip()
        ]))
    
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    
    dataloader_fn = MultiEpochsDataLoader
    train_loader = dataloader_fn(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=workers, worker_init_fn=_worker_init_fn, pin_memory=True, sampler=train_sampler,
        collate_fn=fast_collate, drop_last=True)

    return train_loader, len(train_loader), train_sampler

def get_pytorch_val_loader(data_path, batch_size, workers=5, _worker_init_fn=None, distributed=False):
    valdir = os.path.join(data_path, 'val')
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]))

    if distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        val_sampler = None

    dataloader_fn = MultiEpochsDataLoader
    val_loader = dataloader_fn(
        val_dataset,
        sampler=val_sampler, 
        batch_size=batch_size,
        shuffle=(val_sampler is None),
        num_workers=workers, worker_init_fn=_worker_init_fn, 
        pin_memory=True, collate_fn=fast_collate, drop_last=True)

    return val_loader

if __name__ == '__main__':
    main()
