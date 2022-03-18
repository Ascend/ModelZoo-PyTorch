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
import shutil
from apex import amp

import numpy as np
import random

import torch
import torch.nn.parallel
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import train
from core.function import validate
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import get_model_summary

import dataset
import models
import torch.npu
import os


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
    parser.add_argument('--npu', default=None, type=int, help='NPU id to use.')

    # philly
    parser.add_argument('--distributed', type=str, default=False,
                        help='Use multi-processing distributed training to'
                             'launch N processes per node, which has N NPUs.'
                             'This is the fastest way to use PyTorch for'
                             'either single node or multi node data parallel'
                             'training')
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')
    parser.add_argument('--world-size',
                        default=1,
                        type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--dist-url',
                        default='127.0.0.1',
                        type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--rank',
                        default=0,
                        type=int,
                        help='node rank for distributed training')
    parser.add_argument('--device', default='npu', type=str, help='npu or gpu')
    parser.add_argument('--addr', default='127.0.0.1', type=str, help='master addr')
    parser.add_argument('--amp', default=False, action='store_true',
                        help='use amp to train the model')
    parser.add_argument('--loss-scale', default=None, type=float,
                        help='loss scale using in amp, default -1 means dynamic')
    parser.add_argument('--opt-level', default='O1', type=str,
                        help='loss scale using in amp, default -1 means dynamic')
    parser.add_argument('--device_list', default='0,1,2,3,4,5,6,7', type=str,
                        help='device id list')
    args = parser.parse_args()

    return args


def device_id_to_process_device_map(device_list):
    devices = device_list.split(",")
    devices = [int(x) for x in devices]
    devices.sort()

    process_device_map = dict()
    for process_id, device_id in enumerate(devices):
        process_device_map[process_id] = device_id

    return process_device_map


def main():
    args = parse_args()
    update_config(cfg, args)
    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'train')
    npu = int(os.environ['RANK_ID'])
    if npu == 0:
        logger.info(pprint.pformat(args))
        logger.info(cfg)

    os.environ['LOCAL_DEVICE_ID'] = str(0)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '76472'

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    process_device_map = device_id_to_process_device_map(args.device_list)

    if args.device_list != '':
        npus_per_node = len(process_device_map)
    elif args.device_num > 0:
        npus_per_node = args.device_num
    else:
        npus_per_node = torch.npu.device_count()

    args.world_size = npus_per_node * args.world_size

    args.npu = process_device_map[npu]

    if npu is not None:
        msg = "[npu id:", npu, "]", "Use NPU: {} for training".format(npu)
        logger.info(msg)

    # args.rank = int(os.environ["RANK"])
    args.rank = args.rank * npus_per_node + npu
    # distributed = args.world_size > 1
    distributed = args.distributed
    if distributed:
        if args.device == 'npu':
            dist.init_process_group(backend='hccl', world_size=args.world_size, rank=args.rank)
        else:
            dist.init_process_group(backend='nccl', init_method=args.dist_url,
                                    world_size=args.world_size, rank=args.rank)
    seed = 22
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    calculate_device = 'npu:{}'.format(npu)
    torch.npu.set_device(calculate_device)

    model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
        cfg, is_train=True
    )
    # print(model)
    model = model.to(calculate_device)
    # if not isinstance(model, torch.nn.parallel.DistributedDataParallel):
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[NPU_CALCULATE_DEVICE], broadcast_buffers=False)

    # copy model file
    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, '../lib/models', cfg.MODEL.NAME + '.py'),
        final_output_dir)
    # logger.info(pprint.pformat(model))

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }
    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).to(calculate_device)

    optimizer = get_optimizer(cfg, model)

    if args.amp:
        amp.register_half_function(torch, 'bmm')
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level, loss_scale=args.loss_scale, combine_grad=True)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.npu],
                                                          broadcast_buffers=False)
    else:
        model = torch.nn.DataParallel(model, device_ids=cfg.GPUS)

    print("Data Loading")
    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    train_loader_sampler = None
    if distributed:
        train_loader_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        # val_loader_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=int(cfg.TRAIN.BATCH_SIZE_PER_GPU),
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        sampler=train_loader_sampler
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
        # sampler=val_loader_sampler
    )

    best_perf = 0.0
    best_model = False
    last_epoch = -1

    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth'
    )

    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        if npu == 0: logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file, map_location=calculate_device)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']

        writer_dict['train_global_steps'] = checkpoint['train_global_steps']
        writer_dict['valid_global_steps'] = checkpoint['valid_global_steps']

        model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        if npu == 0:
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(
                checkpoint_file, checkpoint['epoch']))

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg.TRAIN.END_EPOCH, eta_min=cfg.TRAIN.LR_END, last_epoch=last_epoch)

    model.npu()

    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        if distributed:
            train_loader_sampler.set_epoch(epoch)
        if npu == 0: logger.info("=> current learning rate is {:.6f}".format(lr_scheduler.get_last_lr()[0]))
        # train for one epoch
        train(cfg, train_loader, model, criterion, optimizer, epoch,
              final_output_dir, tb_log_dir, writer_dict, is_amp=args.amp, device=calculate_device)
        if npu == 0:
            # evaluate on validation set
            perf_indicator = validate(
                cfg, valid_loader, valid_dataset, model, criterion,
                final_output_dir, tb_log_dir, writer_dict, device=calculate_device
            )

            lr_scheduler.step()

            if perf_indicator >= best_perf:
                best_perf = perf_indicator
                best_model = True
            else:
                best_model = False

            logger.info('=> saving checkpoint to {}'.format(final_output_dir))
            save_checkpoint({
                'epoch': epoch + 1,
                'model': cfg.MODEL.NAME,
                'state_dict': model.state_dict(),
                'best_state_dict': model.module.state_dict(),
                'perf': perf_indicator,
                'optimizer': optimizer.state_dict(),
                'train_global_steps': writer_dict['train_global_steps'],
                'valid_global_steps': writer_dict['valid_global_steps'],
            }, best_model, final_output_dir)
        else:
            lr_scheduler.step()
    if npu == 0:
        final_model_state_file = os.path.join(
            final_output_dir, 'final_state.pth'
        )
        logger.info('=> saving final model state to {}'.format(
            final_model_state_file)
        )
        torch.save(model.module.state_dict(), final_model_state_file)
        writer_dict['writer'].close()


if __name__ == '__main__':
    main()
