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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import argparse
import os
import pprint
import logging
import json

import _init_paths
from core.config import config
from core.config import update_config
from core.function import train_3d, validate_3d
from utils.utils import create_logger
from utils.utils import save_checkpoint, load_checkpoint, load_model_state
from utils.utils import load_backbone_panoptic
import dataset
import models

from apex import amp
import numpy as np
import random
import copy

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--data_path', default=None,
                        help='Dataset root directory path')
    parser.add_argument('--apex', action='store_true',
                        help='Use apex for mixed precision training')
    parser.add_argument('--npu',
                        default=True,
                        help='use NPU')
    parser.add_argument('--num_epochs', default=30, type=int,
                        help='Total train epoch')
    parser.add_argument('--seed', default=1, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--distributed',
                        action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs.')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_rank',
                        default=0,
                        type=int,
                        help='node rank for distributed training')
    parser.add_argument('--addr', default='127.0.0.1', type=str,
                        help='master addr')
    parser.add_argument('--device_id', default=None, type=int,
                        help='device use')

    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    return args


def get_optimizer(model, is_distributed):
    if is_distributed:
        lr = config.TRAIN.LR * 8
    else:
        lr = config.TRAIN.LR

    if model.backbone is not None:
        for params in model.backbone.parameters():
            params.requires_grad = False  # If you want to train the whole model jointly, set it to be True.
    for params in model.root_net.parameters():
        params.requires_grad = True
    for params in model.pose_net.parameters():
        params.requires_grad = True
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    # optimizer = optim.Adam(model.module.parameters(), lr=lr)

    return model, optimizer


def main():
    args = parse_args()

    if args.distributed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        os.environ['PYTHONHASHSEED'] = str(args.seed)

    os.environ['MASTER_ADDR'] = args.addr
    os.environ['MASTER_PORT'] = '29628'

    if args.distributed:
        config.GPUS='0,1,2,3,4,5,6,7'

    gpus = [int(i) for i in config.GPUS.split(',')]


    if args.distributed:
        if 'RANK_SIZE' in os.environ and 'RANK_ID' in os.environ:
            args.rank_size = int(os.environ['RANK_SIZE'])
            args.rank_id = int(os.environ['RANK_ID'])
            args.rank = args.dist_rank * args.rank_size + args.rank_id
            args.world_size = args.world_size * args.rank_size
            args.device_id = args.rank_id
        else:
            raise RuntimeError("init_distributed_mode failed.")
        if args.npu:
            torch.distributed.init_process_group(backend='hccl', init_method="env://",
                                                 world_size=args.world_size, rank=args.rank)
    else:
        if args.device_id is None:
            args.device_id = gpus[0]

    args.is_master_node = not args.distributed or args.device_id == 0

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train', args.is_master_node, args.device_id)

    if args.is_master_node:
        logger.info(pprint.pformat(args))
        logger.info(pprint.pformat(config))

    if args.npu:
        if args.is_master_node:
            print("Use NPU: {} for training".format(gpus))
        device = torch.device('npu:{device_id}'.format(device_id=args.device_id))
        torch.npu.set_device(device)

    if args.is_master_node:
        print('=> Loading data ..')
    if args.data_path is not None:
        config.DATASET.ROOT=args.data_path
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_dataset = eval('dataset.' + config.DATASET.TRAIN_DATASET)(
        config, config.DATASET.TRAIN_SUBSET, True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))

    test_dataset = eval('dataset.' + config.DATASET.TEST_DATASET)(
        config, config.DATASET.TEST_SUBSET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE ,
        shuffle=(train_sampler is None),
        num_workers=config.WORKERS,
        pin_memory=False,
        sampler=train_sampler)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE ,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=False)

    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    if args.is_master_node:
        print('=> Constructing models ..')
    model = eval('models.' + config.MODEL + '.get_multi_person_pose_net')(
        config, is_train=True)

    model.to(device)
    model, optimizer = get_optimizer(model, args.distributed)

    if args.apex:
        if args.npu:
            amp.register_half_function(torch, 'einsum')
        if args.npu and args.distributed:
            model, optimizer = amp.initialize(model, optimizer, opt_level="O1", loss_scale=1)
        else:
            model, optimizer = amp.initialize(model, optimizer, opt_level="O1", loss_scale=None)
        for ls in amp._amp_state.loss_scalers:
            ls._loss_scale = 0.01
            ls._scale_seq_len = 300
        if args.is_master_node:
            logger.info("Use apex for mixed precision training")

    with torch.no_grad():
        if args.npu:
            if args.distributed:
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.device_id],
                                                                  broadcast_buffers=False, find_unused_parameters=True)
            else:
                model = torch.nn.DataParallel(model, device_ids=args.npu).npu(device)

    start_epoch = config.TRAIN.BEGIN_EPOCH
    if args.num_epochs==30:
        end_epoch = config.TRAIN.END_EPOCH
    else:
        end_epoch = start_epoch + args.num_epochs

    best_precision = 0
    if config.NETWORK.PRETRAINED_BACKBONE:
        model = load_backbone_panoptic(model, config.NETWORK.PRETRAINED_BACKBONE)
    if config.TRAIN.RESUME:
        start_epoch, model, optimizer, best_precision = load_checkpoint(model, optimizer, final_output_dir, is_master_node=args.is_master_node, device=device)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    if args.is_master_node:
        print('=> Training...')
    for epoch in range(start_epoch, end_epoch):
        if args.is_master_node:
            print('Epoch: {}'.format(epoch))

        if args.distributed:
            train_sampler.set_epoch(epoch)

        # lr_scheduler.step()
        train_3d(config, model, optimizer, train_loader, epoch, final_output_dir, writer_dict, len(gpus), device=device, is_master_node=args.is_master_node, use_apex=args.apex)
        precision = validate_3d(config, model, test_loader, final_output_dir, device=device, is_master_node=args.is_master_node)

        if precision > best_precision:
            best_precision = precision
            best_model = True
        else:
            best_model = False
        if args.is_master_node:
            logger.info('=> saving checkpoint to {} (Best: {})'.format(final_output_dir, best_model))
        model_copy=copy.deepcopy(model).cpu()
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model_copy.module.state_dict(),
            'precision': best_precision,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)

    final_model_state_file = os.path.join(final_output_dir,
                                          'final_state.pth.tar')
    if args.is_master_node:
        logger.info('saving final model state to {}'.format(
            final_model_state_file))
        logger.info('model best precision is {}'.format(best_precision))
    torch.save(model.module.state_dict(), final_model_state_file)

    writer_dict['writer'].close()

if __name__ == '__main__':
    main()
