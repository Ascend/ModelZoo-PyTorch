# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import argparse
import logging
import os
import cv2
import shutil
import time
import json
import math
import torch
import random
import sys
import apex
import numpy as np
import torch.distributed as dist

from apex import amp

from torch.utils.data import DataLoader
from functools import partial

from utils.log_helper import init_log, print_speed, add_file_handler, Dummy
from utils.load_helper import load_pretrain, restore_from
from utils.average_meter_helper import AverageMeter

from datasets.siam_mask_dataset import DataSets

from utils.lr_helper import build_lr_scheduler
from tensorboardX import SummaryWriter

from utils.config_helper import load_config
from torch.utils.collect_env import get_pretty_env_info

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch Tracking SiamMask Training')

parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--clip', default=10.0, type=float,
                    help='gradient clip value')
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', default='',
                    help='use pre-trained model')
parser.add_argument('--config', dest='config', required=True,
                    help='hyperparameter of SiamMask in json format')
parser.add_argument('--arch', dest='arch', default='', choices=['Custom',],
                    help='architecture of pretrained model')
parser.add_argument('-l', '--log', default="log.txt", type=str,
                    help='log file')
parser.add_argument('-s', '--save_dir', default='snapshot', type=str,
                    help='save dir')
parser.add_argument('--log-dir', default='board', help='TensorBoard log dir')
parser.add_argument('--P', default=1, type=int, choices=[1, 8],
                    help='Device nums.')
parser.add_argument('--num-steps', default=-1, type=int, help='Step time for perfermance.')
parser.add_argument('--seed', default=12345, type=int, help='Set random seed.')

best_acc = 0.


class TimeAverageMeter(object):
    def __init__(self, name, fmt=':f', start_count_index=10):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.start_count_index = start_count_index

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if self.count == 0:
            self.N = n
        self.val = val
        self.count += n
        if self.count > (self.start_count_index * self.N):
            self.sum += val * n
            self.avg = self.sum / (self.count - self.start_count_index * self.N)

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '}({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def collect_env_info():
    env_str = get_pretty_env_info()
    env_str += "\n        OpenCV ({})".format(cv2.__version__)
    return env_str


def worker_init_fn(worker_id, num_workers, rank, seed):
    """Worker init func for dataloader.

    The seed of each worker equals to num_worker * rank + worker_id + user_seed

    """

    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_data_loader(cfg, args):
    logger = logging.getLogger('global')

    logger.info("build train dataset")  # train_dataset
    train_set = DataSets(cfg['train_datasets'], cfg['anchors'], args.epochs)
    train_set.shuffle()

    logger.info("build val dataset")  # val_dataset
    if not 'val_datasets' in cfg.keys():
        cfg['val_datasets'] = cfg['train_datasets']
    val_set = DataSets(cfg['val_datasets'], cfg['anchors'])
    val_set.shuffle()

    init_fn = partial(
        worker_init_fn, num_workers=args.workers, rank=int(os.environ['RANK']), seed=args.seed
    ) if args.seed is not None else None

    # Set distributed sampler for 8P training
    if args.P == 1:
        train_sampler = None
    else:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    
    train_loader = DataLoader(train_set, 
                              batch_size=args.batch, 
                              num_workers=args.workers,
                              shuffle=(train_sampler is None),
                              pin_memory=False, 
                              sampler=train_sampler,
                              worker_init_fn=init_fn,
                              drop_last=True)
    val_loader = DataLoader(val_set, 
                            batch_size=args.batch, 
                            num_workers=args.workers,
                            pin_memory=False, 
                            sampler=None)

    logger.info('build dataset done')
    return train_loader, val_loader, train_sampler


def build_opt_lr(model, cfg, args, epoch):
    backbone_feature = model.features.param_groups(cfg['lr']['start_lr'], cfg['lr']['feature_lr_mult'])
    if len(backbone_feature) == 0:
        trainable_params = model.rpn_model.param_groups(cfg['lr']['start_lr'], cfg['lr']['rpn_lr_mult'], 'mask')
    else:
        trainable_params = backbone_feature + \
                           model.rpn_model.param_groups(cfg['lr']['start_lr'], cfg['lr']['rpn_lr_mult']) + \
                           model.mask_model.param_groups(cfg['lr']['start_lr'], cfg['lr']['mask_lr_mult'])
    
    optimizer = apex.optimizers.NpuFusedSGD(trainable_params, args.lr,
                                            momentum=args.momentum,
                                            weight_decay=args.weight_decay)

    lr_scheduler = build_lr_scheduler(optimizer, cfg['lr'], epochs=args.epochs)

    lr_scheduler.step(epoch)

    return optimizer, lr_scheduler


def main():
    global args, best_acc, tb_writer, logger
    args = parser.parse_args()
    
    # set random seed
    if args.seed:
        seed_everything(args.seed)
    
    # set device
    args.rank = int(os.environ['RANK'])
    args.device = torch.device(f'npu:{args.rank}')
    torch.npu.set_device(args.device)
    
    if args.P > 1:
        args.world_size = int(os.environ['RANK_SIZE'])
        dist.init_process_group(backend='hccl', world_size=args.world_size, rank=args.rank)
    
    args.is_master_node = args.P == 1 or args.rank == 0
    
    if args.is_master_node:
        init_log('global', logging.INFO)
    
        if args.log != "":
            add_file_handler('global', args.log, logging.INFO)
    
        logger = logging.getLogger('global')
        logger.info("\n" + collect_env_info())
        logger.info(args)

    cfg = load_config(args)
    
    if args.is_master_node:
        logger.info("config \n{}".format(json.dumps(cfg, indent=4)))
        tb_writer = None
        # close the tensorboard
        # if args.log_dir:
        #     tb_writer = SummaryWriter(args.log_dir)
        # else:
        #     tb_writer = Dummy()
    else:
        tb_writer = None

    # build dataset
    train_loader, val_loader, train_sampler = build_data_loader(cfg, args)

    if args.arch == 'Custom':
        from models.custom_base import Custom
        model = Custom(pretrain=True, anchors=cfg['anchors'])
    else:
        exit()
        
    if args.is_master_node:
        logger.info(model)

    if args.pretrained:
        model = load_pretrain(model, args.pretrained)
    
    if args.resume and args.start_epoch != 0:
        model.features.unfix((args.start_epoch - 1) / args.epochs)

    model = model.to(args.device)
    optimizer, lr_scheduler = build_opt_lr(model, cfg, args, args.start_epoch)

    # optionally resume from a checkpoint
    if args.resume:
        assert os.path.isfile(args.resume), '{} is not a valid file'.format(args.resume)
        model, optimizer, args.start_epoch, best_acc, arch, checkpoint = restore_from(model, optimizer, args.resume)
    
    # initialize AMP
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1", loss_scale=128.0, combine_grad=True)
    if args.resume:
        amp.load_state_dict(checkpoint['amp'])
    
    # Set DDP for 8P training
    if args.P > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.rank])    
    
    if args.is_master_node:
        logger.info(lr_scheduler)
        logger.info('model prepare done')

    train(train_loader, train_sampler, model, optimizer, lr_scheduler, args.start_epoch, cfg)

def train(train_loader, train_sampler, model, optimizer, lr_scheduler, epoch, cfg):
    global tb_index, best_acc, cur_lr, logger

    if args.seed:
        seed_everything(args.seed)

    cur_lr = lr_scheduler.get_cur_lr()
    
    if args.is_master_node:
        logger = logging.getLogger('global')
    
    avg = AverageMeter()
    FPS_time = TimeAverageMeter('FPS_time', start_count_index=0)
    model.train()
    end = time.time()
    epoch_start_time = time.time()

    def is_valid_number(x):
        return not(math.isnan(x) or math.isinf(x) or x > 1e4)

    num_per_epoch = len(train_loader.dataset) // args.epochs // (args.batch * args.P)
    start_epoch = epoch
    if train_sampler:
        train_loader.sampler.set_epoch(epoch)
        
    for iter, input in enumerate(train_loader):
        if args.num_steps > 0 and iter > args.num_steps:
            return
        if iter % num_per_epoch == 100:
            epoch_start_time = time.time()
        
        # next epoch
        if epoch != iter // num_per_epoch + start_epoch:  
            epoch = iter // num_per_epoch + start_epoch
            FPS_time.update((time.time() - epoch_start_time) / (num_per_epoch - 100))
            epoch_start_time = time.time()
            if args.is_master_node:
                logger.info(f'FPS:{args.batch * args.P / FPS_time.avg}')

            # makedir/save model
            if not os.path.exists(args.save_dir) and args.is_master_node:  
                os.makedirs(args.save_dir)
            
            if args.is_master_node:
                if 'module' not in dir(model):
                    save_checkpoint({
                        'epoch': epoch,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'best_acc': best_acc,
                        'optimizer': optimizer.state_dict(),
                        'anchor_cfg': cfg['anchors'],
                        'amp': amp.state_dict()
                    }, False,
                        os.path.join(args.save_dir, 'checkpoint_e%d.pth' % (epoch)),
                        os.path.join(args.save_dir, 'best.pth'))
                else:
                    save_checkpoint({
                        'epoch': epoch,
                        'arch': args.arch,
                        'state_dict': model.module.state_dict(),
                        'best_acc': best_acc,
                        'optimizer': optimizer.state_dict(),
                        'anchor_cfg': cfg['anchors'],
                        'amp': amp.state_dict()
                    }, False,
                        os.path.join(args.save_dir, 'checkpoint_e%d.pth' % (epoch)),
                        os.path.join(args.save_dir, 'best.pth'))

            if epoch == args.epochs:
                return
            
            if 'module' not in dir(model):
                if model.features.unfix(epoch/args.epochs):
                    if args.is_master_node: 
                        logger.info('unfix part model.')
                    from models.custom_base import Custom
                    model_tmp = Custom(pretrain=False, anchors=cfg['anchors'])
                    model_tmp.to(args.device)
                    model_tmp.load_state_dict(model.state_dict())
                    model_tmp.features.unfix(epoch / args.epochs)
                    model = model_tmp
                    optimizer, lr_scheduler = build_opt_lr(model, cfg, args, epoch)
                    model, optimizer = amp.initialize(model, optimizer, opt_level="O1", loss_scale=128.0,
                                                      combine_grad=True)
                    if args.P > 1:
                        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.rank])
            else:
                if model.module.features.unfix(epoch/args.epochs):
                    if args.is_master_node:
                        logger.info('unfix part model.')
                    from models.custom_base import Custom
                    model_tmp = Custom(pretrain=False, anchors=cfg['anchors'])
                    model_tmp.to(args.device)
                    model_tmp.load_state_dict(model.module.state_dict())
                    model_tmp.features.unfix(epoch / args.epochs)
                    model = model_tmp
                    optimizer, lr_scheduler = build_opt_lr(model, cfg, args, epoch)
                    model, optimizer = amp.initialize(model, optimizer, opt_level="O1", loss_scale=128.0,
                                                      combine_grad=True)
                    if args.P > 1:
                        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.rank])

            lr_scheduler.step(epoch)
            cur_lr = lr_scheduler.get_cur_lr()
           
            if args.is_master_node:
                logger.info('epoch:{}'.format(epoch))

        tb_index = iter
        if iter % num_per_epoch == 0 and iter != 0:
            for idx, pg in enumerate(optimizer.param_groups):
                if args.is_master_node:
                    logger.info("epoch {} lr {}".format(epoch, pg['lr']))
                    if tb_writer:
                        tb_writer.add_scalar('lr/group%d' % (idx+1), pg['lr'], tb_index)

        data_time = time.time() - end

        x = {
            'cfg': cfg,
            'template': torch.autograd.Variable(input[0]).to(args.device),
            'search': torch.autograd.Variable(input[1]).to(args.device),
            'label_cls': torch.autograd.Variable(input[2]).to(args.device),
            'label_loc': torch.autograd.Variable(input[3]).to(args.device),
            'label_loc_weight': torch.autograd.Variable(input[4]).to(args.device),
            'label_mask': torch.autograd.Variable(input[6]).to(args.device),
            'label_mask_weight': torch.autograd.Variable(input[7]).to(args.device),
        }

        outputs = model(x)

        rpn_cls_loss, rpn_loc_loss, rpn_mask_loss = torch.mean(outputs['losses'][0]), \
                                                    torch.mean(outputs['losses'][1]), \
                                                    torch.mean(outputs['losses'][2])
        mask_iou_mean, mask_iou_at_5, mask_iou_at_7 = torch.mean(outputs['accuracy'][0]), \
                                                      torch.mean(outputs['accuracy'][1]), \
                                                      torch.mean(outputs['accuracy'][2])

        cls_weight, reg_weight, mask_weight = cfg['loss']['weight']

        rpn_cls_loss = rpn_cls_loss.to(args.device)
        rpn_loc_loss = rpn_loc_loss.to(args.device)
        rpn_mask_loss = rpn_mask_loss.to(args.device)

        loss = rpn_cls_loss * cls_weight + rpn_loc_loss * reg_weight + rpn_mask_loss * mask_weight

        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        if cfg['clip']['split']:
            torch.nn.utils.clip_grad_norm_(model.module.features.parameters(), cfg['clip']['feature'])
            torch.nn.utils.clip_grad_norm_(model.module.rpn_model.parameters(), cfg['clip']['rpn'])
            torch.nn.utils.clip_grad_norm_(model.module.mask_model.parameters(), cfg['clip']['mask'])
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)  # gradient clip

        if is_valid_number(loss.item()):
            optimizer.step()

        siammask_loss = loss.item()

        batch_time = time.time() - end

        if iter % num_per_epoch > 100:
            avg.update(batch_time=batch_time, data_time=data_time)
        else:
            avg.update(num=0, batch_time=0.)
            avg.update(num=0, data_time=0.)

        avg.update(rpn_cls_loss=rpn_cls_loss, rpn_loc_loss=rpn_loc_loss,
                   rpn_mask_loss=rpn_mask_loss, siammask_loss=siammask_loss,
                   mask_iou_mean=mask_iou_mean, mask_iou_at_5=mask_iou_at_5, mask_iou_at_7=mask_iou_at_7)

        if tb_writer and args.is_master_node:
            tb_writer.add_scalar('loss/cls', rpn_cls_loss, tb_index)
            tb_writer.add_scalar('loss/loc', rpn_loc_loss, tb_index)
            tb_writer.add_scalar('loss/mask', rpn_mask_loss, tb_index)
            tb_writer.add_scalar('mask/mIoU', mask_iou_mean, tb_index)
            tb_writer.add_scalar('mask/AP@.5', mask_iou_at_5, tb_index)
            tb_writer.add_scalar('mask/AP@.7', mask_iou_at_7, tb_index)

        if (iter + 1) % args.print_freq == 0 and args.is_master_node:
            logger.info('Epoch: [{0}][{1}/{2}] lr: {lr:.6f}\t{batch_time:s}\t{data_time:s}'
                        '\t{rpn_cls_loss:s}\t{rpn_loc_loss:s}\t{rpn_mask_loss:s}\t{siammask_loss:s}'
                        '\t{mask_iou_mean:s}\t{mask_iou_at_5:s}\t{mask_iou_at_7:s}'.format(
                        epoch+1, (iter + 1) % num_per_epoch, num_per_epoch, lr=cur_lr, batch_time=avg.batch_time,
                        data_time=avg.data_time, rpn_cls_loss=avg.rpn_cls_loss, rpn_loc_loss=avg.rpn_loc_loss,
                        rpn_mask_loss=avg.rpn_mask_loss, siammask_loss=avg.siammask_loss,
                        mask_iou_mean=avg.mask_iou_mean, mask_iou_at_5=avg.mask_iou_at_5,
                        mask_iou_at_7=avg.mask_iou_at_7))
            print_speed(iter + 1, avg.batch_time.avg, args.epochs * num_per_epoch)
        end = time.time()

def save_checkpoint(state, is_best, filename='checkpoint.pth', best_file='model_best.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_file)


def flush_print(func):
    def new_print(*args, **kwargs):
        func(*args, **kwargs)
        sys.stdout.flush()
    return new_print

print = flush_print(print)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)
    torch.npu.manual_seed_all(seed)


if __name__ == '__main__':
    main()
