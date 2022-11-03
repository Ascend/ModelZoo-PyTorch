# ------------------------------------------------------------------------------
# Copyright 2021 Huawei Technologies Co., Ltd
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
# limitations under the License.)
# ------------------------------------------------------------------------------

import argparse
import os
import pprint
import shutil
import sys

import logging
import time
import timeit
from pathlib import Path

import numpy as np

import apex
from apex import amp
import torch.distributed as dist
import torch.multiprocessing as mp

import torch
if torch.__version__ >= "1.8":
    import torch_npu
import torch.npu
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from tensorboardX import SummaryWriter

import _init_paths
import models
import datasets
from config import config
from config import update_config
from core.criterion import CrossEntropy, OhemCrossEntropy
from core.function import train, validate
from utils.modelsummary import get_model_summary
from utils.utils import create_logger, FullModel


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--seed', type=int, default=304)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('-j', '--workers', default=9, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('--device', default='cuda', type=str, help='npu or gpu')
    parser.add_argument('--eval', '-e', action='store_true', help='evaluate model')
    parser.add_argument('--batch_size', default=3, type=int)
    parser.add_argument('--data_path', default='data/', type=str, help='dataset path')
    parser.add_argument('--resume', action='store_true', help='train resume')
    parser.add_argument('--epoches', default=484, type=int, help='training epoch')
    parser.add_argument('--pretrained', default=False, action='store_true', help='use pretrianed model')
    parser.add_argument('--pth_path', default='', help="load model path")

                
    # distributed training
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument("--local_rank", type=int, default=-1)       
    parser.add_argument('--rank', type=int, default=-1, help='local rank.')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--dist_url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')

    # for Ascend 910 NPU
    parser.add_argument('--addr', default='192.168.88.163',
                        type=str, help='master addr')
    parser.add_argument('--device_list', default='0,1,2,3,4,5,6,7',
                        type=str, help='device id list')
    parser.add_argument('--amp', default=False, action='store_true',
                        help='use amp to train the model')
    parser.add_argument('--loss_scale', default=128.0, type=float,
                        help='loss scale using in amp, default -1 means dynamic')
    parser.add_argument('--opt_level', default='O2', type=str,
                        help='loss scale using in amp, default -1 means dynamic')
    parser.add_argument('--prof', default=False, action='store_true',
                        help='use profiling to evaluate the performance of model')

    args = parser.parse_args()
    update_config(config, args)

    return args

def get_sampler(dataset):
    from utils.distributed import is_distributed
    if is_distributed():
        from torch.utils.data.distributed import DistributedSampler
        return DistributedSampler(dataset)
    else:
        return None

def device_id_to_process_device_map(device_list):
    devices = device_list.split(",")
    devices = [int(x) for x in devices]
    devices.sort()

    process_device_map = dict()
    for process_id, device_id in enumerate(devices):
        process_device_map[process_id] = device_id
    return process_device_map

def profiling(dataloader, device, model, args, optimizer):
    model.train()

    def update(model, images, labels, optimizer):
        losses, _ = model(images, labels)
        loss = losses.mean()
        reduced_loss = loss
        model.zero_grad()
        if args.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

    for i_iter, batch in enumerate(dataloader, 0):
        print(i_iter)
        images, labels, _, _ = batch
        images = images.to(device)         # 3x3x512x1024
        if args.device == 'npu':
            labels = labels.to(device)
        else:
            labels = labels.long().to(device)

        if i_iter < 6:
            update(model, images, labels, optimizer)
        else:
            if args.device == 'npu':
                with torch.autograd.profiler.profile(use_npu=True) as prof:
                    update(model, images, labels, optimizer)
            else:
                with torch.autograd.profiler.profile(use_cuda=True) as prof:
                    update(model, images, labels, optimizer)
            break
    print(prof.key_averages().table(sort_by="self_cpu_time_total"))
    prof.export_chrome_trace('output.prof')

def print_func(inputs, prefix):
    if isinstance(inputs, tuple):
        for i in inputs:
            print_func(i, prefix)
    elif isinstance(inputs, torch.Tensor):
        print(prefix, inputs.shape, inputs.dtype, inputs.storage().npu_format())
    else:
        print(prefix, inputs)

def hook_func(name, module):
    def hook_function(module, inputs, outputs):
        print('================================================')
        print(module)
        print_func(inputs, name + ' inputs')
        print_func(outputs, name + ' outputs')
    return hook_function

def main():
    args = parse_args()
    print('args.local_rank:', args.local_rank)
    
    if args.seed > 0:
        import random
        print('Seeding with', args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)        

    # log
    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    # logger.info(config)

    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    args.process_device_map = device_id_to_process_device_map(args.device_list)
    print('args.process_device_map:', args.process_device_map)

    ngpus_per_node = len(args.process_device_map)
    print('ngpus_per_node:', ngpus_per_node)

    # init_process_group
    args.world_size = args.world_size * ngpus_per_node
    args.distributed = args.world_size > 1
    print('args.world_size:', args.world_size)
    print('args.distributed:', args.distributed)

    os.environ['MASTER_ADDR'] = args.addr
    os.environ['MASTER_PORT'] = '29688'
    if args.distributed:
        dist.init_process_group(backend=args.dist_backend,
                                world_size=args.world_size, rank=args.local_rank)
                                
    # device setting
    if args.device == 'npu':
        device = torch.device('npu:{}'.format(args.local_rank))
        torch.npu.set_device(device)
    if args.device == 'cuda':
        device = torch.device('cuda:{}'.format(args.local_rank))
        torch.cuda.set_device(device)
    print('current device:', device)

    args.batch_size = config.TRAIN.BATCH_SIZE_PER_GPU
    print('batch_size:', args.batch_size)
    print('workers:', args.workers)
    
    # build model
    if args.device == 'npu':
        model = eval('models.'+config.MODEL.NAME +   
                    '.get_seg_model')(config, args.device)
    elif args.device == 'cuda':
        model = eval('models.'+ 'seg_hrnet_ocr_gpu' +   
                    '.get_seg_model')(config, args.device)

    # prepare data
    crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    train_dataset = eval('datasets.'+config.DATASET.DATASET)(  
                        device=device,
                        root=args.data_path,
                        list_path=config.DATASET.TRAIN_SET,
                        num_samples=None,
                        num_classes=config.DATASET.NUM_CLASSES,
                        multi_scale=config.TRAIN.MULTI_SCALE,
                        flip=config.TRAIN.FLIP,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TRAIN.BASE_SIZE,
                        crop_size=crop_size,
                        downsample_rate=config.TRAIN.DOWNSAMPLERATE,
                        scale_factor=config.TRAIN.SCALE_FACTOR)

    if args.distributed:
        train_sampler = get_sampler(train_dataset)
        trainloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers = args.workers,
            # num_workers=6,
            pin_memory=False,
            drop_last=True,
            sampler=train_sampler)
    else:
        trainloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers = args.workers,
            pin_memory=False)
            
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_dataset = eval('datasets.'+config.DATASET.DATASET)(
                        device=device,
                        root=args.data_path,
                        list_path=config.DATASET.TEST_SET,
                        num_samples=config.TEST.NUM_SAMPLES,
                        num_classes=config.DATASET.NUM_CLASSES,
                        multi_scale=False,
                        flip=False,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TEST.BASE_SIZE,
                        crop_size=test_size,
                        downsample_rate=1)

    if args.distributed:
        test_sampler = get_sampler(test_dataset)
        testloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=(test_sampler is None),
            num_workers=args.workers,
            # num_workers=6,
            pin_memory=False,
            sampler=test_sampler)
    else:
        testloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=False)
    print("data is ready!!!")

    # criterion
    criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                weight=train_dataset.class_weights)

    model = FullModel(model, criterion).to(device)


    # optimizer
    if config.TRAIN.OPTIMIZER == 'sgd':

        params_dict = dict(model.named_parameters())
        if config.TRAIN.NONBACKBONE_KEYWORDS:
            bb_lr = []
            nbb_lr = []
            nbb_keys = set()
            for k, param in params_dict.items():
                if any(part in k for part in config.TRAIN.NONBACKBONE_KEYWORDS):
                    nbb_lr.append(param)
                    nbb_keys.add(k)
                else:
                    bb_lr.append(param)
            print(nbb_keys)
            params = [{'params': bb_lr, 'lr': config.TRAIN.LR}, {'params': nbb_lr, 'lr': config.TRAIN.LR * config.TRAIN.NONBACKBONE_MULT}]
        else:
            params = [{'params': list(params_dict.values()), 'lr': config.TRAIN.LR}]

        if args.amp and args.device == 'npu':
            optimizer = apex.optimizers.NpuFusedSGD(params,
                                    lr=config.TRAIN.LR,
                                    momentum=config.TRAIN.MOMENTUM,
                                    weight_decay=config.TRAIN.WD,
                                    nesterov=config.TRAIN.NESTEROV,
                                    )
        else:
            optimizer = torch.optim.SGD(params,
                                    lr=config.TRAIN.LR,
                                    momentum=config.TRAIN.MOMENTUM,
                                    weight_decay=config.TRAIN.WD,
                                    nesterov=config.TRAIN.NESTEROV,
                                    )
    else:
        raise ValueError('Only Support SGD optimizer')
    

    # Apex
    if args.amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level,loss_scale=args.loss_scale, combine_grad=True)
        print("==========using apex")
    
    
    # DDP
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            find_unused_parameters=True,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False
        )
        print("=======using DDP===========")
    else:
        print('Using GPU {} for training'.format(args.device_list))


    epoch_iters = np.int(train_dataset.__len__() / 
                        config.TRAIN.BATCH_SIZE_PER_GPU / len(args.process_device_map))
        
    if args.pretrained:
        model_state_file = args.pth_path
        if os.path.isfile(model_state_file):
            print("=> loading best model best.pth from {}".format(model_state_file))
            pretrained_dict = torch.load(model_state_file, map_location={'npu:0': 'cpu'})
            if 'state_dict' in pretrained_dict:
                pretrained_dict = pretrained_dict['state_dict']
            model.model.load_state_dict({k.replace('model.', ''): v for k, v in pretrained_dict.items() if k.startswith('model.')})

    # resume
    best_mIoU = 0
    last_epoch = 0
    if args.resume:
        model_state_file = os.path.join(final_output_dir,
                                        'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            if args.device == 'npu':
                checkpoint = torch.load(model_state_file, map_location={'npu:0': 'cpu'})
            else:
                checkpoint = torch.load(model_state_file, map_location={'cuda:0': 'cpu'})
            best_mIoU = checkpoint['best_mIoU']
            last_epoch = checkpoint['epoch']
            dct = checkpoint['state_dict']
        
            if args.distributed:
                model.module.model.load_state_dict({k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('model.')})
            else:
                model.model.load_state_dict({k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('model.')})
                
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})"
                        .format(checkpoint['epoch']))
        if args.distributed:
            torch.distributed.barrier()

    if args.eval:
        valid_loss, mean_IoU, IoU_array = validate(config, device, 
                    args, testloader, model, writer_dict)
        return
 
    if args.prof:
        profiling(trainloader, device, model, args, optimizer)
        return 

    start = timeit.default_timer()
    end_epoch = args.epoches
    num_iters = args.epoches * epoch_iters
    
    for epoch in range(last_epoch, end_epoch):

        current_trainloader = trainloader
        if current_trainloader.sampler is not None and hasattr(current_trainloader.sampler, 'set_epoch'):
            current_trainloader.sampler.set_epoch(epoch)

        train(config, device, args, epoch, args.epoches, 
                epoch_iters, config.TRAIN.LR, num_iters,
                trainloader, optimizer, model, writer_dict)

        valid_loss, mean_IoU, IoU_array = validate(config, device, 
                    args, testloader, model, writer_dict)

        if not args.distributed or (args.distributed and args.local_rank == 0):
            logger.info('=> saving checkpoint to {}'.format(
                final_output_dir + 'checkpoint.pth.tar'))
            torch.save({
                'epoch': epoch+1,
                'best_mIoU': best_mIoU,
                'state_dict': model.module.state_dict() if args.distributed else model.state_dict(), 
                'optimizer': optimizer.state_dict(),
            }, os.path.join(final_output_dir,'checkpoint.pth.tar'))
            if mean_IoU > best_mIoU:
                best_mIoU = mean_IoU
                torch.save(model.module.state_dict() if args.distributed else model.state_dict(),
                        os.path.join(final_output_dir, 'best.pth'))
            msg = 'Loss: {:.3f}, MeanIU: {: 4.4f}, Best_mIoU: {: 4.4f}'.format(
                        valid_loss, mean_IoU, best_mIoU)
            logging.info(msg)
            logging.info(IoU_array)

    if not args.distributed or (args.distributed and args.local_rank == 0):

        torch.save(model.module.state_dict() if args.distributed else model.state_dict(),
                os.path.join(final_output_dir, 'final_state.pth'))

        writer_dict['writer'].close()
        end = timeit.default_timer()
        logger.info('Hours: %d' % np.int((end-start)/3600))
        logger.info('Done')


if __name__ == '__main__':
    main()

    
