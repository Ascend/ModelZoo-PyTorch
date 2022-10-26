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

##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2020
##
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import sys
import time
import json
import logging
import argparse

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel
if torch.__version__ >="1.8":
    import torch_npu

import apex
from apex import amp

from module.config import get_cfg
from module.models.build import get_model
from module.datasets import get_dataset
from module.transforms import get_transform
from module.loss import get_criterion
from module.utils import (save_checkpoint, accuracy, AverageMeter, LR_Scheduler,
                          torch_dist_sum, mkdir, cached_log_stream, PathManager)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


##############################################################################
# Class
##############################################################################
class Options():
    def __init__(self):
        # data settings
        parser = argparse.ArgumentParser(description='ResNeSt Training')
        parser.add_argument('--config-file', type=str, default=None,
                            help='training configs')
        parser.add_argument('--outdir', type=str, default='output',
                            help='output directory')
        parser.add_argument('--logtxt', default='log_apex.txt', type=str,
                            help='output log')

        # checking point
        parser.add_argument('--resume', type=str, default=None,
                            help='put the path to resuming file if needed')

        # distributed
        parser.add_argument('--world-size', default=1, type=int,
                            help='number of nodes for distributed training')
        parser.add_argument('--rank', default=0, type=int,
                            help='node rank for distributed training')
        parser.add_argument('--dist-backend', default='nccl', type=str,
                            help='distributed backend', choices=['nccl', 'hccl'])

        # evaluation option
        parser.add_argument('--eval-only', action='store_true', default=False,
                            help='evaluating')
        parser.add_argument('--export', type=str, default=None,
                            help='put the path to resuming file if needed')

        # apex
        parser.add_argument('--amp', default=False, action='store_true',
                            help='use amp to train the model')
        parser.add_argument('--loss-scale', default="dynamic",
                            help='loss scale using in amp, default is dynamic')
        parser.add_argument('--opt-level', default='O2', type=str,
                            help='train mode in amp', choices=['O0', 'O1', 'O2', 'O3'])

        # Ascend
        parser.add_argument('--device', default='npu', type=str,
                            help='device for training', choices=['gpu', 'npu'])
        parser.add_argument('--addr', default='127.0.0.1', type=str,
                            help='master addr')
        parser.add_argument('--port', default='29688', type=str,
                            help='master port')
        parser.add_argument('--prof', default=False, action='store_true',
                            help='use profiling to evaluate the performance of model')

        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        return args


###############################################################################
# Variables
###############################################################################
# global variable
best_pred = 0.0
acclist_train = []
acclist_val = []


##############################################################################
# Functions
##############################################################################
def profiling(data_loader, model, criterion, optimizer, args):
    # switch to train mode
    model.train()

    ######################################################################
    # update
    def update(model, images, target, optimizer):
        output = model(images)
        loss = criterion(output, target)
        if args.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.zero_grad()
        optimizer.step()
    ######################################################################

    for step, (images, target) in enumerate(data_loader):
        loc = 'npu:{}'.format(args.gpu)
        images = images.to(loc, non_blocking=True).to(torch.float)
        target = target.to(torch.int32).to(loc, non_blocking=True)

        if step < 5:
            update(model, images, target, optimizer)
        else:
            with torch.autograd.profiler.profile(use_npu=True) as prof:
                update(model, images, target, optimizer)
            prof.export_chrome_trace("output.prof")


def main():
    # init the global
    global best_pred, acclist_train, acclist_val
    # load args
    args = Options().parse()
    args.distributed = args.world_size > 1

    # load config
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.OPTIMIZER.LR = cfg.OPTIMIZER.LR * args.world_size

    # set address & port
    os.environ['MASTER_ADDR'] = args.addr
    os.environ['MASTER_PORT'] = args.port

    # set seed
    if cfg.SEED is not None:
        torch.manual_seed(cfg.SEED)
        torch.cuda.manual_seed(cfg.SEED)

    args.gpu = args.rank

    # init device
    print(torch.__version__)
    device_loc = 'npu:%d' % args.rank

    if args.gpu == 0 or args.world_size == 1:
        mkdir(args.outdir)
        filename = os.path.join(args.outdir, args.logtxt)
        fh = logging.StreamHandler(cached_log_stream(filename))
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
        plain_formatter = logging.Formatter(
            "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S")
        fh.setFormatter(plain_formatter)
        logger.info(args)
        print(args)
    
    if args.distributed:
        logger.info('rank: {} / {}'.format(args.rank, args.world_size))
        print('rank: {} / {}'.format(args.rank, args.world_size))

    # init process group
    if args.distributed:
        dist.init_process_group(backend=args.dist_backend,
                                # init_method=args.dist_url,
                                world_size=args.world_size,
                                rank=args.rank)

    # set device
    torch.npu.set_device(device_loc)

    # init dataloader
    transform_train, transform_val = get_transform(cfg.DATA.DATASET)(cfg.DATA.BASE_SIZE,
                                                                     cfg.DATA.CROP_SIZE,
                                                                     cfg.DATA.RAND_AUG)
    trainset = get_dataset(cfg.DATA.DATASET)(root=cfg.DATA.ROOT,
                                             transform=transform_train,
                                             train=True)
    valset = get_dataset(cfg.DATA.DATASET)(root=cfg.DATA.ROOT,
                                           transform=transform_val,
                                           train=False)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(trainset,
                                               batch_size=cfg.TRAINING.BATCH_SIZE,
                                               shuffle=(train_sampler is None),
                                               num_workers=cfg.TRAINING.WORKERS,
                                               pin_memory=True,
                                               sampler=train_sampler)

    if args.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(valset, shuffle=False)
    else:
        val_sampler = None
    val_loader = torch.utils.data.DataLoader(valset,
                                             batch_size=cfg.TRAINING.TEST_BATCH_SIZE,
                                             shuffle=False,
                                             num_workers=cfg.TRAINING.WORKERS,
                                             pin_memory=True,
                                             sampler=val_sampler)

    # set device & init the model
    model_kwargs = {}
    model_kwargs['num_classes'] = cfg.MODEL.NUM_CLASSES
    if cfg.MODEL.FINAL_DROP > 0.0:
        model_kwargs['final_drop'] = cfg.MODEL.FINAL_DROP
    if cfg.TRAINING.LAST_GAMMA:
        model_kwargs['last_gamma'] = True

    model = get_model(cfg.MODEL.NAME)(**model_kwargs)

    model = model.to(device_loc)

    # criterion and optimizer
    criterion, train_loader = get_criterion(cfg, train_loader, device_loc)
    criterion = criterion.to(device_loc)

    if cfg.OPTIMIZER.DISABLE_BN_WD:
        parameters = model.named_parameters()
        param_dict = {}
        for k, v in parameters:
            param_dict[k] = v
        bn_params = [v for n, v in param_dict.items() if ('bn' in n or 'bias' in n)]
        rest_params = [v for n, v in param_dict.items() if not ('bn' in n or 'bias' in n)]
        if args.gpu == 0 or args.world_size == 1:
            logger.info(" Weight decay NOT applied to BN parameters ")
            print(" Weight decay NOT applied to BN parameters ")
            logger.info('len(parameters): {} = {} + {}'
                        .format(len(list(model.parameters())), len(bn_params), len(rest_params)))
            print('len(parameters): {} = {} + {}'
                        .format(len(list(model.parameters())), len(bn_params), len(rest_params)))
        optimizer = apex.optimizers.NpuFusedSGD([{'params': bn_params, 'weight_decay': 0},
                                                 {'params': rest_params, 'weight_decay': cfg.OPTIMIZER.WEIGHT_DECAY}],
                                                lr=cfg.OPTIMIZER.LR,
                                                momentum=cfg.OPTIMIZER.MOMENTUM,
                                                weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY)
    else:
        optimizer = apex.optimizers.NpuFusedSGD(model.parameters(),
                                                lr=cfg.OPTIMIZER.LR,
                                                momentum=cfg.OPTIMIZER.MOMENTUM,
                                                weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY)

    # amp initialize for model and optimizer
    if args.amp:
        amp.register_float_function(torch, 'sigmoid')
        amp.register_float_function(torch, 'softmax')
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level, loss_scale=args.loss_scale, combine_grad=True)

    # model DDP
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.gpu])

    # check point
    if args.resume is not None:
        if os.path.isfile(args.resume):
            if args.gpu == 0 or args.world_size == 1:
                logger.info("=> loading checkpoint '{}'".format(args.resume))
                print("=> loading checkpoint '{}'".format(args.resume))
            with PathManager.open(args.resume, "rb") as f:
                checkpoint = torch.load(f, map_location=device_loc)
            cfg.TRAINING.START_EPOCHS = checkpoint['epoch'] + 1 if cfg.TRAINING.START_EPOCHS == 0 \
                else cfg.TRAINING.START_EPOCHS
            best_pred = checkpoint['best_pred']
            acclist_train = checkpoint['acclist_train']
            acclist_val = checkpoint['acclist_val']
            model.module.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if args.amp:
                amp.load_state_dict(checkpoint['amp'])
            if args.gpu == 0 or args.world_size == 1:
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
                print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            raise RuntimeError("=> no resume checkpoint found at '{}'".format(args.resume))

    # benchmark
    cudnn.benchmark = True

    scheduler = LR_Scheduler(cfg.OPTIMIZER.LR_SCHEDULER,
                             base_lr=cfg.OPTIMIZER.LR,
                             num_epochs=cfg.TRAINING.EPOCHS,
                             iters_per_epoch=len(train_loader),
                             warmup_epochs=cfg.OPTIMIZER.WARMUP_EPOCHS)

    ######################################################################
    # train
    def train(epoch):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        model.train()

        # FPS
        fps = 0
        data_time = AverageMeter()
        data_start_time = time.time()
        batch_time = AverageMeter()
        available_time = AverageMeter()

        # Loss & Accuracy
        losses = AverageMeter()
        top1 = AverageMeter()

        # Best Acc & Acc List
        global best_pred, acclist_train

        for batch_idx, (data, target) in enumerate(train_loader):
            data_cost_time = time.time() - data_start_time
            data_time.update(data_cost_time)
            batch_start_time = time.time()

            scheduler(optimizer, batch_idx, epoch, best_pred)
            if not cfg.DATA.MIXUP:
                data = data.to(device_loc, non_blocking=True).to(torch.float)
                target = target.to(device_loc, non_blocking=True).to(torch.int32)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            # loss backward with amp
            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            optimizer.step()

            # update time, accuracy and loss
            if not cfg.DATA.MIXUP:
                acc1 = accuracy(output, target, topk=(1,))
                top1.update(acc1[0], data.size(0))

            losses.update(loss.item(), data.size(0))

            batch_cost_time = time.time() - batch_start_time
            batch_time.update(batch_cost_time)

            if batch_idx > 1:
                available_time.update(batch_cost_time + data_cost_time)
                fps = cfg.TRAINING.BATCH_SIZE * args.world_size / (available_time.avg)

            # show info
            if batch_idx % 100 == 0 and (args.gpu == 0 or args.world_size == 1):
                if cfg.DATA.MIXUP:
                    logger.info('Batch: %d| Lr: %.6f | FPS: %.2f | Loss: %.3f' %
                    (batch_idx, optimizer.state_dict()['param_groups'][0]['lr'], fps, losses.avg))
                    print('Batch: %d| Lr: %.6f | FPS: %.2f | Loss: %.3f' %
                    (batch_idx, optimizer.state_dict()['param_groups'][0]['lr'], fps, losses.avg))
                else:
                    logger.info('Batch: %d| Lr: %.6f | FPS: %.2f | Loss: %.3f | Top1: %.3f' %
                    (batch_idx, optimizer.state_dict()['param_groups'][0]['lr'], fps, losses.avg, top1.avg))
                    print('Batch: %d| Lr: %.6f | FPS: %.2f | Loss: %.3f | Top1: %.3f' %
                    (batch_idx, optimizer.state_dict()['param_groups'][0]['lr'], fps, losses.avg, top1.avg))

            # start loading dataloader
            data_start_time = time.time()

        acclist_train += [top1.avg]
    ######################################################################

    ######################################################################
    # validate
    def validate(epoch):
        model.eval()
        top1 = AverageMeter()
        top5 = AverageMeter()
        global best_pred, acclist_train, acclist_val
        is_best = False
        for batch_idx, (data, target) in enumerate(val_loader):
            data = data.to(device_loc, non_blocking=True).to(torch.float)
            target = target.to(device_loc, non_blocking=True).to(torch.int32)
            with torch.no_grad():
                output = model(data)
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                top1.update(acc1[0], data.size(0))
                top5.update(acc5[0], data.size(0))

        # sum all
        if args.distributed:
            sum1, cnt1, sum5, cnt5 = torch_dist_sum(args.gpu, args.device, top1.sum, top1.count, top5.sum, top5.count)
            top1_acc = torch.true_divide(sum(sum1), sum(cnt1))
            top5_acc = torch.true_divide(sum(sum5), sum(cnt5))
        else:
            top1_acc = top1.sum / top1.count
            top5_acc = top5.sum / top5.count

        if args.gpu == 0 or args.world_size == 1:
            logger.info('Validation: Top1: %.3f | Top5: %.3f' % (top1_acc, top5_acc))
            print('Validation: Top1: %.3f | Top5: %.3f' % (top1_acc, top5_acc))
            if args.eval_only:
                return top1_acc, top5_acc

            # save checkpoint
            acclist_val += [top1_acc]
            if top1_acc > best_pred:
                best_pred = top1_acc
                is_best = True
            if args.amp:
                save_checkpoint({'epoch': epoch,
                                 'state_dict': model.module.state_dict()
                                                if args.distributed else model.state_dict(),
                                 'optimizer': optimizer.state_dict(),
                                 'amp': amp.state_dict(),
                                 'best_pred': best_pred,
                                 'acclist_train': acclist_train,
                                 'acclist_val': acclist_val},
                                directory=args.outdir,
                                is_best=False,
                                filename='checkpoint_apex_{}.pth'.format(epoch))
            else:
                save_checkpoint({'epoch': epoch,
                                 'state_dict': model.module.state_dict()
                                                if args.distributed else model.state_dict(),
                                 'optimizer': optimizer.state_dict(),
                                 'best_pred': best_pred,
                                 'acclist_train': acclist_train,
                                 'acclist_val': acclist_val},
                                directory=args.outdir,
                                is_best=False,
                                filename='checkpoint_{}.pth'.format(epoch))
        return top1_acc.item(), top5_acc.item()
    ######################################################################

    if args.prof:
        profiling(train_loader, model, criterion, optimizer, args)
        return

    if args.export:
        if args.gpu == 0 or args.world_size == 1:
            with PathManager.open(args.export + '.pth', "wb") as f:
                torch.save(model.module.state_dict(), f)
        return

    if args.eval_only:
        top1_acc, top5_acc = validate(cfg.TRAINING.START_EPOCHS)
        metrics = {"top1": top1_acc,
                   "top5": top5_acc}
        if args.gpu == 0 or args.world_size == 1:
            with PathManager.open(os.path.join(args.outdir, 'metrics.json'), "w") as f:
                json.dump(metrics, f)
        return

    for epoch in range(cfg.TRAINING.START_EPOCHS, cfg.TRAINING.EPOCHS):
        tic = time.time()
        train(epoch)
        if epoch % 10 == 0:
            top1_acc, top5_acc = validate(epoch)
        elapsed = time.time() - tic
        if args.gpu == 0 or args.world_size == 1:
            logger.info('Epoch: {}, Time cost: {}'.format(epoch, elapsed))
            print('Epoch: {}, Time cost: {}'.format(epoch, elapsed))

    # final evaluation
    top1_acc, top5_acc = validate(cfg.TRAINING.START_EPOCHS - 1)

    # save final checkpoint
    if args.gpu == 0 or args.world_size == 1:
        if args.amp:
            save_checkpoint({
                'epoch': cfg.TRAINING.EPOCHS - 1,
                'state_dict': model.module.state_dict()
                               if args.distributed else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'amp': amp.state_dict(),
                'best_pred': best_pred,
                'acclist_train': acclist_train,
                'acclist_val': acclist_val,
            },
                directory=args.outdir,
                is_best=False,
                filename='checkpoint_apex_final.pth')
        else:
            save_checkpoint({
                'epoch': cfg.TRAINING.EPOCHS - 1,
                'state_dict': model.module.state_dict()
                               if args.distributed else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_pred': best_pred,
                'acclist_train': acclist_train,
                'acclist_val': acclist_val,
            },
                directory=args.outdir,
                is_best=False,
                filename='checkpoint_final.pth')

        # save final model weights
        metrics = {"top1": top1_acc,
                   "top5": top5_acc}
        if args.amp:
            model_weights_name = 'model_weights_apex.pth'
            metrics_name = 'metrics_apex.json'
        else:
            model_weights_name = 'model_weights.pth'
            metrics_name = 'metrics.json'

        with PathManager.open(os.path.join(args.outdir, model_weights_name), "wb") as f:
            if args.distributed:
                torch.save(model.module.state_dict(), f)
            else:
                torch.save(model.state_dict(), f)
        with PathManager.open(os.path.join(args.outdir, metrics_name), "w") as f:
            json.dump(metrics, f)

if __name__ == "__main__":
    main()
