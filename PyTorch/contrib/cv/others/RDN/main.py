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
import copy
import datetime
import os
import time
import sys
import torch
import torch.utils.data
from torch import nn

import torch.npu  # not for GPU
import random
import numpy as np
import utils

from models import RDN
from datasets import TrainDataset, EvalDataset
import argparse
import torch.multiprocessing as mp
from utils import AverageMeter

try:
    import apex
    from apex import amp
except ImportError:
    amp = None


def flush_print(func):
    def new_print(*args, **kwargs):
        func(*args, **kwargs)
        sys.stdout.flush()
    return new_print
print = flush_print(print)

def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args):
    model.train()
    batchtime = AverageMeter(start_count_index=3)
    cond = 4000 // (args.batch_size * args.gpus)

    for i, (image, target) in enumerate(data_loader):
        start_time = time.time()
        image, target = image.to(device), target.to(device)  # for RDN
        output = model(image)
        loss = criterion(output, target)

        optimizer.zero_grad()
        if args.apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        batch_size = image.shape[0]
        if i < cond:
            batchtime.update(time.time() - start_time)

        if args.is_master_node:
            print("Epoch {} step {},loss :{},img/s :{},time :{}".format(epoch, i, loss, batch_size / batchtime.val,
                                                                        batchtime.val))

    if args.is_master_node:
        print('epoch:{} FPS: {:.3f}'.format(epoch, args.gpus * args.batch_size / batchtime.avg))


def evaluate(model, criterion, data_loader, device, args):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, args, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            epoch_psnr = utils.calc_psnr(output, target, args.scale, device)
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['epoch_psnr'].update(epoch_psnr.item(), n=batch_size)  # RDN

    metric_logger.synchronize_between_processes(device)

    print('epoch_psnr={} '.format(metric_logger.epoch_psnr.global_avg))

    return metric_logger.epoch_psnr.global_avg


def _get_cache_path(filepath):
    import hashlib
    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def main(args):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29688'

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    if args.apex:
        if sys.version_info < (3, 0):
            raise RuntimeError("Apex currently only supports Python 3. Aborting.")
        if amp is None:
            raise RuntimeError("Failed to import apex. Please install apex from https://www.github.com/nvidia/apex "
                               "to enable mixed-precision training.")

    if args.output_dir:
        utils.mkdir(args.output_dir)

    

    if args.distributed:  # ********************************************
        mp.spawn(main_worker, nprocs=args.gpus, args=(args,))
       
    else:
        main_worker(args.gpus, args)


def main_worker(nprocs, args):
    local_rank = 0
    if args.distributed:
        torch.distributed.init_process_group(backend="hccl",
                                             init_method='env://',
                                             world_size=args.world_size * args.gpus,
                                             rank=nprocs)
        local_rank = torch.distributed.get_rank()
    args.is_master_node = not args.distributed or local_rank == 0
    if args.is_master_node:
        print(args)
    args.device_id = args.device_id + local_rank
    print('device_id=', args.device_id)
    device = torch.device(f'npu:{args.device_id}')  # npu
    torch.npu.set_device(device)  # for npu
   

    # Data loading code
    print("Loading data")
    traindir = os.path.join(args.data_path, 'train')
    valdir = os.path.join(args.data_path, 'val')

    print("Loading training data")
    st = time.time()
    cache_path = _get_cache_path(traindir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print("Loading dataset_train from {}".format(cache_path))
        dataset, _ = torch.load(cache_path)
    else:

        dataset = TrainDataset(args.train_file, patch_size=args.patch_size, scale=args.scale)
        if args.is_master_node and args.cache_dataset:
            print("Saving dataset_train to {}".format(cache_path))
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset, traindir), cache_path)
    print("Took", time.time() - st)

    print("Loading validation data")
    cache_path = _get_cache_path(valdir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print("Loading dataset_test from {}".format(cache_path))
        dataset_test, _ = torch.load(cache_path)
    else:
        dataset_test = EvalDataset(args.eval_file)
        if args.is_master_node and args.cache_dataset:
            print("Saving dataset_test to {}".format(cache_path))
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset_test, valdir), cache_path)

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, pin_memory=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, num_workers=0, pin_memory=True)

    if args.is_master_node:
        print("Creating model")
    model = RDN(scale_factor=args.scale,
                num_channels=3,
                num_features=args.num_features,
                growth_rate=args.growth_rate,
                num_blocks=args.num_blocks,
                num_layers=args.num_layers)
    if args.pretrained:
        checkpoint = torch.load(args.pretrained_weight_path, map_location='cpu')
        if 'module.' in list(checkpoint['model'].keys())[0]:
            checkpoint['model'] = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
        model.load_state_dict(checkpoint['model'], strict=False)

    model.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = nn.L1Loss().to(device)

    optimizer = apex.optimizers.NpuFusedAdam(

        model.parameters(), lr=args.lr)

    if args.apex:
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=args.apex_opt_level,
                                          loss_scale=args.loss_scale_value,
                                          combine_grad=True)  # conbine_grad?

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
        evaluate(model, criterion, data_loader_test, device=device, args=args)  ############
        return

    if args.is_master_node:
        print("Start training")

    best_weights = copy.deepcopy(model_without_ddp.state_dict())
    best_epoch = 0
    best_psnr = 0.0

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args)

        lr_scheduler.step()
        epoch_psnr = evaluate(model, criterion, data_loader_test, device=device, args=args)

        if (epoch + 1) % 40 == 0 and args.is_master_node and args.output_dir:
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args}
            utils.save_on_master(
                checkpoint,
                os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
            utils.save_on_master(
                checkpoint,
                os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.is_master_node and epoch_psnr > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr
            best_weights = copy.deepcopy(model_without_ddp.state_dict())
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if args.is_master_node:
        print('Training time {}'.format(total_time_str))
        print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
        if args.epochs > 100:
            checkpoint = {
                'model': best_weights,
                'psnr': best_psnr,
                'epoch': best_epoch,
                'args': args}
            utils.save_on_master(
                checkpoint,
                os.path.join(args.output_dir, 'best.pth'))


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    parser.add_argument('--data_path', default='', help='dataset')
    parser.add_argument('--model', default='resnet18', help='model')
    parser.add_argument('--device_id', default=0, type=int, help='device id')
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')

    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr_step_size', default=600, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr_gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='outputs', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true"
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true"
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true"
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true"
    )
    parser.add_argument(
        '--pretrained_weight_path',
        default='',
        help='pretrained weight path')

    # Mixed precision training parameters
    parser.add_argument(
        '--apex',
        action='store_true',
        help='Use apex for mixed precision training')
    parser.add_argument(
        '--apex-opt-level',
        default='O1',
        type=str,
        help='For apex mixed precision training' 'O0 for FP32 training, O1 for mixed precision training.'
             'For further detail, see https://github.com/NVIDIA/apex/tree/master/examples/imagenet'
    )
    parser.add_argument(
        '--loss_scale_value',
        default=1024,
        type=int,
        help='set loss scale value.')

    # distributed training parameters
    parser.add_argument(
        '--world-size',
        default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs.')
    parser.add_argument('--dist_rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument('--seed',
                        default=123, type=int, help='Manually set random seed')
    # RDN training parameters
    parser.add_argument('--weights-file', type=str)
    parser.add_argument('--num-features', type=int, default=64)
    parser.add_argument('--growth-rate', type=int, default=64)
    parser.add_argument('--num-blocks', type=int, default=16)
    parser.add_argument('--num-layers', type=int, default=8)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--patch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--gpus', type=int, default=1, help='num of gpus of per node')
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--eval_file', type=str, required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
