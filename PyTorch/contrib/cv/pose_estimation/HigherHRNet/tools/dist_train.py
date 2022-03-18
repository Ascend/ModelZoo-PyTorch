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
# ============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import random
import shutil
import sys
import warnings
import logging
import time
import threading


import torch
import torch.npu
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from apex import amp
from tensorboardX import SummaryWriter

import _init_paths
import models

from config import cfg
from config import update_config
from core.loss import MultiLossFactory
from dataset import make_dataloader
from utils.utils import create_logger
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import setup_logger
from utils.utils import AverageMeter
from utils.vis import save_debug_images

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # distributed training
    parser.add_argument('--gpu',
                        help='gpu id for multiprocessing training',
                        type=str)
    parser.add_argument('--world-size',
                        default=1,
                        type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--dist-url',
                        default='tcp://127.0.0.1:23456',
                        type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--rank',
                        default=0,
                        type=int,
                        help='node rank for distributed training')

    parser.add_argument('--device', default='npu', type=str, help='npu or gpu')
    parser.add_argument('--device_num', default=-1, type=int,
                        help='device_num')
    parser.add_argument('--device-list', default='', type=str, help='device id list')
    parser.add_argument('--addr', default='10.136.181.115', type=str, help='master addr')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    # apex
    parser.add_argument('--amp', default=False, action='store_true',
                        help='use amp to train the model')
    parser.add_argument('--loss-scale', default=None, type=float,
                        help='loss scale using in amp, default -1 means dynamic')
    parser.add_argument('--opt-level', default='O1', type=str,
                        help='loss scale using in amp, default -1 means dynamic')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    cfg.defrost()
    cfg.RANK = args.rank
    cfg.freeze()

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'train'
    )

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29688'

    logger.info(pprint.pformat(args))
    logger.info(cfg)
    os.environ['LOCAL_DEVICE_ID'] = str(0)
    print("+++++++++++++++++++++++++++LOCAL_DEVICE_ID:", os.environ['LOCAL_DEVICE_ID'])
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or cfg.MULTIPROCESSING_DISTRIBUTED
    # print("world_size",args.world_size)
    ngpus_per_node = int(os.getenv("RANK_SIZE"))
    if cfg.MULTIPROCESSING_DISTRIBUTED:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(
            main_worker,
            nprocs=ngpus_per_node,
            args=(ngpus_per_node, args, final_output_dir, tb_log_dir)
        )
    else:
        # Simply call main_worker function
        main_worker(
            ','.join([str(i) for i in cfg.GPUS]),
            ngpus_per_node,
            args,
            final_output_dir,
            tb_log_dir
        )

def main_worker(
        gpu, ngpus_per_node, args, final_output_dir, tb_log_dir
):

    if args.device_list != '':
        args.gpu = int(args.device_list.split(',')[gpu])
    else:
        args.gpu = gpu

    print("[npu id:", args.gpu, "]", "++++++++++++++++ before set LOCAL_DEVICE_ID:", os.environ['LOCAL_DEVICE_ID'])
    os.environ['LOCAL_DEVICE_ID'] = str(args.gpu)
    print("[npu id:", args.gpu, "]", "++++++++++++++++ LOCAL_DEVICE_ID:", os.environ['LOCAL_DEVICE_ID'])

    if args.gpu is not None:
        print("[npu id:", args.gpu, "]", "Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if cfg.MULTIPROCESSING_DISTRIBUTED:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        print('Init process group: dist_url: {}, world_size: {}, rank: {}'.
              format(args.dist_url, args.world_size, args.rank))
        if args.device == 'npu':
            dist.init_process_group(backend=cfg.DIST_BACKEND,# init_method=args.dist_url,
                                    world_size=args.world_size, rank=args.rank)
        else:
            dist.init_process_group(backend=cfg.DIST_BACKEND, init_method=args.dist_url,
                                    world_size=args.world_size, rank=args.rank)
    update_config(cfg, args)

    # setup logger
    logger, _ = setup_logger(final_output_dir, args.rank, 'train')
    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=True
    )
    # copy model file
    if not cfg.MULTIPROCESSING_DISTRIBUTED or (
            cfg.MULTIPROCESSING_DISTRIBUTED
            and args.rank % ngpus_per_node == 0
    ):
        this_dir = os.path.dirname(__file__)
        shutil.copy2(
            os.path.join(this_dir, '../lib/models', cfg.MODEL.NAME + '.py'),
            final_output_dir
        )

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    if not cfg.MULTIPROCESSING_DISTRIBUTED or (
            cfg.MULTIPROCESSING_DISTRIBUTED
            and args.rank % ngpus_per_node == 0
    ):
        dump_input = torch.rand(
            (1, 3, cfg.DATASET.INPUT_SIZE, cfg.DATASET.INPUT_SIZE)
        )

    loc = 'npu:{}'.format(args.gpu)
    torch.npu.set_device(loc)
    model = model.to(loc)
    # define loss function (criterion) and optimizer
    loss_factory = MultiLossFactory(cfg).to(loc)
   
    best_perf = -1
    best_model = False
    last_epoch = -1
    optimizer = get_optimizer(cfg, model)

    if args.amp:
        amp.register_half_function(torch, 'bmm')
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level, loss_scale=args.loss_scale, combine_grad=True)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], broadcast_buffers=False)
    train_loader,train_loader_sampler= make_dataloader(
        cfg, is_train=True, distributed=args.distributed
    )

    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth.tar')
    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'], strict=False)

        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=last_epoch)

    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        # train one epoch
        if train_loader_sampler is not None:
            train_loader_sampler.set_epoch(epoch)
        do_train(cfg, args, model, train_loader, loss_factory, optimizer, epoch,
                 final_output_dir, tb_log_dir, writer_dict)
        lr_scheduler.step()

        perf_indicator = epoch
        if perf_indicator >= best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        if not cfg.MULTIPROCESSING_DISTRIBUTED or (
                cfg.MULTIPROCESSING_DISTRIBUTED
                and args.rank == 0
        ):
            logger.info('=> saving checkpoint to {}'.format(final_output_dir))
            if not cfg.MULTIPROCESSING_DISTRIBUTED:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model': cfg.MODEL.NAME,
                    'state_dict': model.state_dict(),
                    'best_state_dict': model.state_dict(),
                    'perf': perf_indicator,
                    'optimizer': optimizer.state_dict(),
                }, best_model, final_output_dir)
            else:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model': cfg.MODEL.NAME,
                    'state_dict': model.state_dict(),
                    'best_state_dict': model.module.state_dict(),
                    'perf': perf_indicator,
                    'optimizer': optimizer.state_dict(),
                }, best_model, final_output_dir)

    final_model_state_file = os.path.join(
        final_output_dir, 'final_state{}.pth.tar'.format(gpu)
    )

    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    torch.save(model.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


def do_train(cfg, args, model, data_loader, loss_factory, optimizer, epoch,
             output_dir, tb_log_dir, writer_dict):
    logger = logging.getLogger("Training")

    batch_time = AverageMeter()
    data_time = AverageMeter()

    heatmaps_loss_meter = [AverageMeter() for _ in range(cfg.LOSS.NUM_STAGES)]
    push_loss_meter = [AverageMeter() for _ in range(cfg.LOSS.NUM_STAGES)]
    pull_loss_meter = [AverageMeter() for _ in range(cfg.LOSS.NUM_STAGES)]

    loc = 'npu:{}'.format(args.gpu)
    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, heatmaps, masks, joints) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)
#        with torch.autograd.profiler.profile(use_npu=True) as prof:
        images = images.to(loc, non_blocking=True)
        heatmaps = list(map(lambda x: x.to(loc, non_blocking=True), heatmaps))
        masks = list(map(lambda x: x.to(loc, non_blocking=True), masks))
        joints = list(map(lambda x: x.to(loc, non_blocking=True), joints))

        # compute output
        outputs = model(images)


        # loss = loss_factory(outputs, heatmaps, masks)
        heatmaps_losses, push_losses, pull_losses = \
            loss_factory(outputs, heatmaps, masks, joints)

        loss = 0
        for idx in range(cfg.LOSS.NUM_STAGES):
            if heatmaps_losses[idx] is not None:
                heatmaps_loss = heatmaps_losses[idx].mean(dim=0)
                heatmaps_loss_meter[idx].update(
                    heatmaps_loss.item(), images.size(0)
                )
                loss = loss + heatmaps_loss
                if push_losses[idx] is not None:
                    push_loss = push_losses[idx].mean(dim=0)
                    push_loss_meter[idx].update(
                        push_loss.item(), images.size(0)
                    )
                    loss = loss + push_loss
                if pull_losses[idx] is not None:
                    pull_loss = pull_losses[idx].mean(dim=0)
                    pull_loss_meter[idx].update(
                        pull_loss.item(), images.size(0)
                    )
                    loss = loss + pull_loss

        # compute gradient and do update step
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

        if i % cfg.PRINT_FREQ == 0 and cfg.RANK == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed: {speed:.1f} samples/s\t' \
                  'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  '{heatmaps_loss}{push_loss}{pull_loss}'.format(
                      epoch, i, len(data_loader),
                      batch_time=batch_time,
                      speed=images.size(0)/batch_time.val,
                      data_time=data_time,
                      heatmaps_loss=_get_loss_info(heatmaps_loss_meter, 'heatmaps'),
                      push_loss=_get_loss_info(push_loss_meter, 'push'),
                      pull_loss=_get_loss_info(pull_loss_meter, 'pull')
                  )
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            for idx in range(cfg.LOSS.NUM_STAGES):
                writer.add_scalar(
                    'train_stage{}_heatmaps_loss'.format(i),
                    heatmaps_loss_meter[idx].val,
                    global_steps
                )
                writer.add_scalar(
                    'train_stage{}_push_loss'.format(idx),
                    push_loss_meter[idx].val,
                    global_steps
                )
                writer.add_scalar(
                    'train_stage{}_pull_loss'.format(idx),
                    pull_loss_meter[idx].val,
                    global_steps
                )
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            for scale_idx in range(len(outputs)):
                prefix_scale = prefix + '_output_{}'.format(
                    cfg.DATASET.OUTPUT_SIZE[scale_idx]
                )
                save_debug_images(
                    cfg, images, heatmaps[scale_idx], masks[scale_idx],
                    outputs[scale_idx], prefix_scale
                )



def _get_loss_info(loss_meters, loss_name):
    msg = ''
    for i, meter in enumerate(loss_meters):
        msg += 'Stage{i}-{name}: {meter.val:.3e} ({meter.avg:.3e})\t'.format(
            i=i, name=loss_name, meter=meter
        )

    return msg

if __name__ == '__main__':
    main()
