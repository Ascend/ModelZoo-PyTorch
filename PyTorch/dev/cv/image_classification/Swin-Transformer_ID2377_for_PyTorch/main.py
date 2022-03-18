#
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
#
import os
import torch
import time
import torch.nn as nn
import argparse
import datetime
import torch.distributed as dist
import numpy as np
import torch.backends.cudnn as cudnn
from config import get_config
from models import build_model
from data import build_loader
from logger import create_logger
from optimizer import build_optimizer
from lr_scheduler import build_scheduler
from utils import load_checkpoint, get_grad_norm, save_checkpoint, reduce_tensor
from timm.utils import AverageMeter, accuracy
import torch.npu
import os
import apex
from apex import amp
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')

def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script')
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path of config file')
    parser.add_argument('--opts', nargs='+', help="Modify config options by adding 'KEY VALUE' pairs ")

    # easy hyper-params config
    parser.add_argument('--batch-size', type=int, metavar='N', help='batch size for single GPU')
    parser.add_argument('--data-path', type=str, metavar='PATH', help='path of dataset')
    parser.add_argument('--resume', type=str, metavar='PATH', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, metavar='N', help='gradient accumulation steps')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag>')
    parser.add_argument('--tag', type=str, help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')

    # distributed training
    parser.add_argument('--local_rank', type=int, required=True, help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()
    config = get_config(args)

    return args, config

def main(config):

    dataset_train, dataset_valid, data_loader_train, data_loader_valid = build_loader(config)
    logger.info("=> Creating model '{}/{}'".format(config.MODEL.TYPE, config.MODEL.NAME))
    model = build_model(config)
    model.npu()
    logger.info(str(model))

    optimizer = build_optimizer(config, model)
    
    model, optimizer = amp.initialize(model, optimizer, opt_level='O2',combine_grad=True)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))
    criterion = nn.CrossEntropyLoss().npu()

    max_accuracy = 0.0

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model, optimizer, lr_scheduler, logger)
        if config.EVAL_MODE:
            return

    logger.info("=> start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)
        train(config, model, criterion, data_loader_train, optimizer, lr_scheduler, epoch)
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, model, optimizer, lr_scheduler, logger, epoch, max_accuracy)

        acc1, acc5, loss = validate(config, data_loader_valid, model, criterion)
        logger.info("Accuracy of the network on the {} test images: {:.1f}%".format(len(dataset_valid), acc1))
        max_accuracy = max(max_accuracy, acc1)
        logger.info("Max accuracy: {:.2f}".format(max_accuracy))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

def train(config, model, criterion, data_loader, optimizer, lr_scheduler, epoch):
    model.train()
    optimizer.zero_grad()
    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for i, (samples, targets) in enumerate(data_loader):
        if i > 100:
           pass
        start_time = time.time()
        samples = samples.npu(non_blocking=True)
        targets = targets.npu(non_blocking=True)

        outputs = model(samples)

        if config.TRAIN.ACCUMULATION_STEPS > 1:
            loss = criterion(outputs, targets)
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            #loss.backward()
            if config.TRAIN.CLIP_GRAD:
                gard_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
            else:
                gard_norm = get_grad_norm(model.parameters())
            if (i + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + i)
        else:
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            #loss.backward()
            if config.TRAIN.CLIP_GRAD:
                gard_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
            else:
                gard_norm = get_grad_norm(model.parameters())
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + i)
        torch.npu.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        norm_meter.update(gard_norm)
        batch_time.update(time.time() - end)
        end = time.time()
        step_time = end - start_time
        FPS = config.DATA.BATCH_SIZE / step_time
        if i % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            etas = batch_time.avg * (num_steps - i)
            logger.info(
                f'Train: [{epoch+1}/{config.TRAIN.EPOCHS}][{i+1}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'time/step(s):{step_time:.3f}\t'
                f'FPS:{FPS:.3f}\t'
            )
    epoch_time = time.time() - start
    logger.info("EPOCH {} training takes {}".format(epoch+1, datetime.timedelta(seconds=int(epoch_time))))


@torch.no_grad()
def validate(config, data_loader, model, criterion):
    model.eval()
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    for i, (images, targets) in enumerate(data_loader):
        if i > 100:
           pass
        images = images.npu(non_blocking=True)
        targets = targets.npu(non_blocking=True)

        output = model(images)

        loss = criterion(output, targets)
        acc1, acc5 = accuracy(output, targets, topk=(1, 5))

        acc1, acc5, loss = reduce_tensor(acc1), reduce_tensor(acc5), reduce_tensor(loss)
        loss_meter.update(loss.item(), targets.size(0))
        acc1_meter.update(acc1.item(), targets.size(0))
        acc5_meter.update(acc5.item(), targets.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            logger.info(
                f'Test: [{i+1}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
            )
    logger.info(' * Acc@1 {:.3f} Acc@5 {:.3f}'.format(acc1_meter.avg, acc5_meter.avg))

    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg

if __name__ == '__main__':

    _, config = parse_option()

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        print("RANK and WORLD_SIZE in environ: '{}/{}'".format(rank, world_size))
    else:
        rank = -1
        world_size = -1
    torch.npu.set_device(config.LOCAL_RANK)
    dist.init_process_group(backend='hccl', init_method='env://', world_size=world_size, rank=rank)
    dist.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=config.MODEL.NAME)
    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info("Full config saved to '{}'".format(path))

    logger.info(config.dump())

    main(config)
