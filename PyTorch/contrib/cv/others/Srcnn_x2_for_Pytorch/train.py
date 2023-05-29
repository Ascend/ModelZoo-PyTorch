# -*- coding: utf-8 -*-
#
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

import datetime
import os
import time
import sys

import torch
if torch.__version__ >= '1.8':
    import torch_npu
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from skimage.metrics import peak_signal_noise_ratio

if torch.__version__ >= '1.8':
    import torch_npu
import torch.npu
import random
import numpy as np
import copy

import utils
from utils import AverageMeter
from models import SRCNN
from datasets import TrainDataset, EvalDataset

try:
    import apex
    from apex import amp
except ImportError:
    amp = None


def train_one_epoch(model, criterion, optimizer, train_dataloader, device, epoch, args):
    model.train()
    epoch_losses = AverageMeter()
    batch_time = AverageMeter()
    train_psnr = AverageMeter()

    for i, data in enumerate(train_dataloader):
        
        if i > 5 :
            start_time = time.time()

        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        preds = model(inputs)
        preds = torch_npu.npu_format_cast(preds, 2)
        loss = criterion(preds, labels)
        epoch_losses.update(loss.item(), len(inputs))

        preds = preds.clamp(0.0, 1.0)
        preds = preds.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        train_psnr.update(peak_signal_noise_ratio(preds, labels), len(inputs))

        optimizer.zero_grad()

        if args.apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()

        if i > 5 :
            batch_time.update(time.time() - start_time)

    return train_psnr.avg, epoch_losses.avg, batch_time.avg

def evaluate(model, criterion, eval_dataloader, device):
    model.eval()
    eval_psnr = AverageMeter()

    for i, data in enumerate(eval_dataloader):
        
        inputs, labels = data
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.no_grad():
            preds = model(inputs).clamp(0.0, 1.0)
        
        preds = torch_npu.npu_format_cast(preds, 2)
        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()
        eval_psnr.update(peak_signal_noise_ratio(preds, labels), len(inputs))

    return eval_psnr.avg

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = args.lr * (0.1 ** (epoch // (args.epochs//3 - 3)))

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

def profiling(data_loader, model, criterion, optimizer, device, args):
    # switch to train mode
    model.train()

    def update(model, inputs, labels, optimizer):
        output = model(inputs)
        output = torch_npu.npu_format_cast(output, 2)
        loss = criterion(output, labels)
        if args.apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.zero_grad()
        optimizer.step()

    for step, (inputs, labels) in enumerate(data_loader):

        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if step < 5:
            update(model, inputs, labels, optimizer)
        else:
            with torch.autograd.profiler.profile(use_npu=True) as prof:
                update(model, inputs, labels, optimizer)
            break

    prof.export_chrome_trace(args.prof_path)


def main(args):

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '54265'

    cudnn.benchmark = True
    torch.manual_seed(args.seed)

    if args.apex:
        if sys.version_info < (3, 0):
            raise RuntimeError("Apex currently only supports Python 3. Aborting.")
        if amp is None:
            raise RuntimeError("Failed to import apex. Please install apex from https://www.github.com/nvidia/apex "
                               "to enable mixed-precision training.")
    
    args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    if args.distributed:
        rank_size = utils.init_distributed_mode(args)

    args.is_master_node = not args.distributed or args.device_id == 0
    if args.is_master_node:
        print(args)

    device = torch.device(f'npu:{args.device_id}')
    torch.npu.set_device(device)

    # Data loading code
    if args.is_master_node:
        print("Loading data")
    st = time.time()
    train_dataset = TrainDataset(args.train_file)
    if args.is_master_node:
        print("Took", time.time() - st)

    eval_dataset = EvalDataset(args.eval_file)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=(train_sampler is None),
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True,
                                  sampler=train_sampler)

    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    if args.is_master_node:
        print("Creating model")

    model = SRCNN()

    if args.pretrained:
        if os.path.exists(args.pretrained_weight_path):
            checkpoint = torch.load(args.pretrained_weight_path, map_location='cpu')
            if 'module.' in list(checkpoint['model'].keys())[0]:
                checkpoint['model'] = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
            model.load_state_dict(checkpoint['model'], strict=False)
            if args.is_master_node:
                print("Use pretrained model")
    elif args.resume:
        if args.is_master_node:
            print("Continue Training")
    else:
        if args.is_master_node:
            print("Train from beginning")

    model.to(device)

    criterion = nn.MSELoss().to(device)

    #optimizer = optim.Adam([
    optimizer = apex.optimizers.NpuFusedAdam([
        {'params': model.conv1.parameters()},
        {'params': model.conv2.parameters()},
        {'params': model.conv3.parameters(), 'lr': args.lr * 0.1}
    ], lr=args.lr)

    if args.apex:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.apex_opt_level, loss_scale=args.loss_scale_value, combine_grad=True)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.device_id], broadcast_buffers=False)
        model_without_ddp = model.module

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    if args.test_only:
        test_psnr = evaluate(model, criterion, eval_dataloader, device=device)
        if args.is_master_node:
            print('test_psnr={:.2f}' .format(test_psnr))
        return

    if args.prof:
        profiling(train_dataloader, model, criterion, optimizer, device, args)
        return

    if args.is_master_node:
        print("Start training")

    start_time = time.time()

    best_checkpoint = copy.deepcopy({'model': model_without_ddp.state_dict()})

    best_epoch = 0
    best_psnr = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        
        if args.distributed:
            train_sampler.set_epoch(epoch)

        if args.warm_up:
            adjust_learning_rate(optimizer, epoch, args)

        train_psnr, train_loss, batch_time = train_one_epoch(model, criterion, optimizer, train_dataloader, device, epoch, args)
        lr_scheduler.step()
        eval_psnr = evaluate(model, criterion, eval_dataloader, device=device)
        
        if args.distributed: 
            if args.device_id == 0:
                fps = args.batch_size * rank_size / batch_time
                print('Epoch:[{}/{}]\t train_psnr={:.2f}\t train_loss={:.6f}\t train_time={:.4f}\t fps={:.4f}\t eval_psnr={:.2f}\t LearningRate={:.6f}' \
                .format(epoch, args.epochs, train_psnr, train_loss, batch_time, fps, eval_psnr, get_lr(optimizer)))
        else:
            fps = args.batch_size / batch_time
            print('Epoch:[{}/{}]\t train_psnr={:.2f}\t train_loss={:.6f}\t train_time={:.4f}\t fps={:.4f}\t eval_psnr={:.2f}\t LearningRate={:.6f}' \
            .format(epoch, args.epochs, train_psnr, train_loss, batch_time, fps, eval_psnr, get_lr(optimizer)))
      
        if args.is_master_node and args.outputs_dir and epoch % 10 == 0:
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args}
            torch.save(
                checkpoint,
                os.path.join(args.outputs_dir, 'model_{}.pth'.format(epoch)))
        
        if args.is_master_node:
            if eval_psnr > best_psnr:
                best_epoch = epoch
                best_loss = train_loss
                best_fps = fps
                best_psnr = eval_psnr
                best_checkpoint = copy.deepcopy({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args})
    
    if args.is_master_node:
        print('best epoch: {}\t train_loss={:.6f}\t fps={:.4f}\t eval_psnr={:.2f}\t'.format(best_epoch, best_loss, best_fps, best_psnr))
        torch.save(best_checkpoint, os.path.join(args.outputs_dir, 'best.pth'))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if args.is_master_node:
        print('Training time {}'.format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch SRCNN Training')

    parser.add_argument('--train-file', type=str, required=True)
    parser.add_argument('--eval-file', type=str, required=True)
    parser.add_argument('--outputs-dir', type=str, required=True)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--device_id', default=0, type=int, help='device id')
    parser.add_argument('-b', '--batch-size', default=16, type=int)
    parser.add_argument('--epochs', default=400, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-j', '--num_workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 8)')
    parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')
    parser.add_argument('--lr-step-size', default=200, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--seed', default=123, type=int, help='Manually set random seed')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument("--test-only", dest="test_only", help="Only test the model", action="store_true", )
    parser.add_argument("--pretrained", dest="pretrained", help="Use pre-trained models", action="store_true", )
    parser.add_argument('--pretrained_weight_path', default='', help='pretrained weight path')
    parser.add_argument('--prof', default=False, action='store_true',
                        help='use profiling to evaluate the performance of pretrainedmodels')
    parser.add_argument('--prof_path', default="", help='prof path')
    parser.add_argument('--warm_up', dest='warm_up', action='store_true', help='warm up')
    parser.add_argument('--warm_up_epochs', default=10, type=int, help='warm up epochs')

    # Mixed precision training parameters
    parser.add_argument('--apex', action='store_true', help='Use apex for mixed precision training')
    parser.add_argument('--apex-opt-level', default='O1', type=str,
                        help='For apex mixed precision training'
                             'O0 for FP32 training, O1 for mixed precision training.'
                             'For further detail, see https://github.com/NVIDIA/apex/tree/master/examples/imagenet'
                        )
    parser.add_argument('--loss_scale_value', default=128.0, type=float, help='set loss scale value.')

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                        'N processes per node, which has N GPUs.')
    parser.add_argument('--dist_rank', default=0, type=int, help='node rank for distributed training')

    args = parser.parse_args()

    return args

# for servers to immediately record the logs
def flush_print(func):
    def new_print(*args, **kwargs):
        func(*args, **kwargs)
        sys.stdout.flush()
    return new_print

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

if __name__ == "__main__":
    
    args = parse_args()
    print = flush_print(print)
    curtime = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))  
    print("Current time = ", curtime)
    main(args)
