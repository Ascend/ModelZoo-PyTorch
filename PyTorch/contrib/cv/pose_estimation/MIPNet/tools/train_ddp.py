#coding:utf-8
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from apex.optimizers import NpuFusedAdam
import torch.distributed as dist
import torchvision.transforms as transforms
import argparse  ###
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
from apex import amp ###
from apex.parallel import convert_syncbn_model ###
from apex.parallel import DistributedDataParallel as DDP ###

import dataset
import models

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12345'

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    # philly
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
    ###
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--is_distributed', type=int, default=1)
    parser.add_argument('--perf', type=int, default=0)
    parser.add_argument('--check_point', type=str, default="")
  
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    print(args.local_rank)
    loc = "npu:{}".format(args.local_rank)
    world_size = 8 if args.is_distributed else 1
    torch.npu.set_device(args.local_rank) ###
    torch.distributed.init_process_group(  
        'hccl',
        init_method='env://',
        world_size=world_size,
        rank=args.local_rank
    ) ###
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'train')
    if dist.get_rank() == 0:
        logger.info(pprint.pformat(args))
        logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    print('DEBUG')

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=True
    )

    # copy model file
    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, '../lib/models', cfg.MODEL.NAME + '.py'),
        final_output_dir)
    if dist.get_rank() == 0:
        logger.info(pprint.pformat(model))

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    dump_input = torch.rand(
        (1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0])
    )
    # if dist.get_rank() == 0:
    #     logger.info(get_model_summary(model, dump_input))

    if cfg.TEST.MODEL_FILE:
        if dist.get_rank() == 0:
            logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        if dist.get_rank() == 0:
            print('no model weights given, training with random weights!')

    #model = torch.nn.DataParallel(model).npu() 
    # REW: // 通过这个数据并行DP，数据会根据多少个gpu划分多少份，所以下面乘了len(gpu)
    model = model.npu()

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).npu()

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_dataset = eval('dataset.'+cfg.DATASET.TRAIN_DATASET)(
        cfg=cfg, image_dir=cfg.DATASET.TRAIN_IMAGE_DIR, annotation_file=cfg.DATASET.TRAIN_ANNOTATION_FILE, \
        dataset_type=cfg.DATASET.TRAIN_DATASET_TYPE, \
        image_set=cfg.DATASET.TRAIN_SET, is_train=True, \
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )

    valid_dataset = eval('dataset.'+cfg.DATASET.TEST_DATASET)(
        cfg=cfg, image_dir=cfg.DATASET.TEST_IMAGE_DIR, annotation_file=cfg.DATASET.TEST_ANNOTATION_FILE, \
        dataset_type=cfg.DATASET.TEST_DATASET_TYPE, \
        image_set=cfg.DATASET.TEST_SET, is_train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    # REW:DDP 通过sampler自动管理去划分数据，所以只需要传单p数据量就够了
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # ----------------------------------------------
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=(train_sampler is None),
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        sampler=train_sampler, #if training else None,
        # drop_last=True 
    )#sampler\drop_last 为新加
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    # # # ----------------------------------------------

    best_perf = 0.0
    best_model = False
    last_epoch = -1
    # optimizer = get_optimizer(cfg, model)
    optimizer = NpuFusedAdam(model.parameters(), lr=cfg.TRAIN.LR)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    
    #checkpoint_file = "output/coco/pose_hrnet/w48_384x288_adam_lr1e-3/checkpoint_151.pth"
    checkpoint_file = args.check_point
    ###
    #sfjj
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1", loss_scale="dynamic", combine_grad=True)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False)
    
    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        print("exec checkpoint sfjj")
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file, map_location=loc)
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=last_epoch
    )
    #torch.npu.set_start_fuzz_compile_step(4)
    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        train_sampler.set_epoch(epoch) ###
        
        # # train for one epoch
        if dist.get_rank() == 0:
            print('training on coco')
        s_t = time.time()
        train(args, cfg, train_loader, model, criterion, optimizer, epoch,
              final_output_dir, tb_log_dir, writer_dict)
        if dist.get_rank() == 0:
            print("sfjj 1 train: {}".format(time.time() - s_t))
        lr_scheduler.step()

        if epoch % cfg.EPOCH_EVAL_FREQ == 0 and not args.perf:
        # if epoch % 1 == 0:
            ### evaluate on validation set
            #sfjj
            if dist.get_rank() == 0:
                vs_t = time.time()
                perf_indicator = validate(
                    cfg, valid_loader, valid_dataset, model, criterion,
                    final_output_dir, tb_log_dir, writer_dict, epoch=epoch, print_prefix='baseline'
                )
                print("sfjj 1 validate: {}".format(time.time() - vs_t))
        else:
            perf_indicator = 0.0

        # perf_indicator = 0.0
        if dist.get_rank() == 0:
            if perf_indicator >= best_perf:
                best_perf = perf_indicator
                best_model = True
            else:
                best_model = False
            if epoch % 20 ==0:
                logger.info('=> saving checkpoint to {}'.format(final_output_dir))
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model': cfg.MODEL.NAME,
                    'state_dict': model.state_dict(),
                    'latest_state_dict': model.module.state_dict(),
                    'best_state_dict': model.module.state_dict(),
                    'perf': perf_indicator,
                    'optimizer': optimizer.state_dict(),
                }, best_model, final_output_dir, filename='checkpoint_{}.pth'.format(epoch + 1))
        dist.barrier()
    # # ----------------------------------------------
    ## validate as ending point
    
    if dist.get_rank()==0 and not args.perf:
        print('validate on coco')
        validate(cfg, valid_loader, valid_dataset, model, criterion,
                final_output_dir, tb_log_dir, writer_dict)

        # # ----------------------------------------------
        if dist.get_rank() == 0:
            final_model_state_file = os.path.join(
                final_output_dir, 'final_state.pth'
            )
            logger.info('=> saving final model state to {}'.format(
                final_model_state_file)
            )
            torch.save(model.module.state_dict(), final_model_state_file)
            writer_dict['writer'].close()


if __name__ == '__main__':
    option = {}
    option["ACL_OP_COMPILER_CACHE_MODE"] = "enable"
    kernel_dir = "./kernel_meta"
    if not os.path.exists(kernel_dir):
        os.mkdir(kernel_dir)
    option["ACL_OP_COMPILER_CACHE_DIR"] = kernel_dir
    print("option:",option)
    torch.npu.set_option(option)
    main()
