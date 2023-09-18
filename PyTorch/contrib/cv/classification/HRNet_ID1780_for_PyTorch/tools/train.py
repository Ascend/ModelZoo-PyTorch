# Copyright (c) Soumith Chintala 2016,
# All rights reserved
#
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://spdx.org/licenses/BSD-3-Clause.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil
import sys

import torch
if torch.__version__ >= "1.8":
    import torch_npu


import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import _init_paths
import models
from config import config
from config import update_config
from core.function import train
from core.function import validate
from utils.modelsummary import get_model_summary
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from apex import amp


def parse_args():
    parser = argparse.ArgumentParser(description='Train classification network')

    parser.add_argument('data_path',
                        help='data path',
                        type=str)

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('--resume',
                        help='pth file',
                        type=str,
                        default='')
    parser.add_argument('--pretrained',
                        help='load pth',
                        type=str,
                        default='')

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
    parser.add_argument('--testModel',
                        help='testModel',
                        type=str,
                        default='')

    parser.add_argument('--addr',
                        help='addr',
                        type=str,
                        default='')

    parser.add_argument('--device_id',
                        help='device_id',
                        type=int,
                        default='')

    parser.add_argument('--bs',
                        help='batch_size',
                        type=int,
                        default='')

    parser.add_argument('--lr',
                        help='lr',
                        type=float,
                        default='')

    parser.add_argument('--dn',
                        help='device_num',
                        type=str,
                        default='')
    
    parser.add_argument('--nproc',
                        help='nproc',
                        type=int,
                        default='')

    parser.add_argument('--train_epochs',
                        help='train epochs',
                        type=int,
                        default='')
    parser.add_argument('--stop_step',
                        help='stop_step',
                        type=bool,
                        default=False)

    args = parser.parse_args()
    update_config(config, args)

    return args


def main():
    if os.getenv('ALLOW_FP32', False) and os.getenv('ALLOW_HF32', False):
        raise RuntimeError('ALLOW_FP32 and ALLOW_HF32 cannot be set at the same time!')
    elif os.getenv('ALLOW_HF32', False):
        torch.npu.conv.allow_hf32 = True
    elif os.getenv('ALLOW_FP32', False):
        torch.npu.conv.allow_hf32 = False
        torch.npu.matmul.allow_hf32 = False
    args = parse_args()

    host_ip = args.addr
    device_id = args.device_id
    device_num = int(args.dn)
    
    os.environ['MASTER_ADDR'] = host_ip
    os.environ['MASTER_PORT'] = '29988'
    rank_id = device_id

    if device_num > 1:
        dist.init_process_group(backend='hccl', rank=rank_id, world_size=device_num)
    
    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')
    
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    if args.pretrained:
        model = eval('models.' + config.MODEL.NAME + '.get_cls_net')(
            config)
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint)
    else:
        model = eval('models.' + config.MODEL.NAME + '.get_cls_net')(
            config)

    dump_input = torch.rand(
        (1, 3, config.MODEL.IMAGE_SIZE[1], config.MODEL.IMAGE_SIZE[0])
    )
    logger.info(get_model_summary(model, dump_input))    

    # copy model file
    this_dir = os.path.dirname(__file__)
    models_dst_dir = os.path.join(final_output_dir, 'models')
    if os.path.exists(models_dst_dir):
        shutil.rmtree(models_dst_dir)
    shutil.copytree(os.path.join(this_dir, '../lib/models'), models_dst_dir)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    loc = 'npu:{}'.format(device_id)
    torch.npu.set_device(loc)
    model = model.npu()
  
    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().npu()
    lr = args.lr * device_num
    optimizer = get_optimizer(config, model, lr)
    if not os.getenv('ALLOW_FP32') and not os.getenv('ALLOW_HF32'):
        model, optimizer = amp.initialize(model, optimizer, opt_level="O2", loss_scale=128.0)

    if device_num > 1:    
        model = DDP(model, device_ids=[device_id])

    best_perf = 0.0
    best_model = False
    last_epoch = config.TRAIN.BEGIN_EPOCH
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir,
                                        'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file)
            last_epoch = checkpoint['epoch']
            best_perf = checkpoint['perf']
            model.module.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})"
                        .format(checkpoint['epoch']))
            best_model = True
            
    if isinstance(config.TRAIN.LR_STEP, list):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR,
            last_epoch-1
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR,
            last_epoch-1
        )

    # Data loading code
    data_path = args.data_path
    traindir = os.path.join(data_path, "train")
    valdir = os.path.join(data_path, "val")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(config.MODEL.IMAGE_SIZE[0]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    )

    if device_num > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    nproc = args.nproc
    num_workers = nproc // device_num
    bs = args.bs
    kwargs = {"pin_memory_device": "npu"} if torch.__version__ >= "2.0" else {}
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=False,
        **kwargs
    )
    valid_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(int(config.MODEL.IMAGE_SIZE[0] / 0.875)),
            transforms.CenterCrop(config.MODEL.IMAGE_SIZE[0]),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=bs,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        **kwargs
    )

    train_epochs = args.train_epochs
    for epoch in range(train_epochs):
        if device_num > 1:
            train_sampler.set_epoch(epoch)
        lr_scheduler.step()
        # train for one epoch
        train(config, train_loader, model, criterion, optimizer, epoch,
              final_output_dir, tb_log_dir, writer_dict, device_num, bs, args.stop_step)
        # evaluate on validation set
        perf_indicator = validate(config, valid_loader, model, criterion,
                                  final_output_dir, tb_log_dir, writer_dict)
        if perf_indicator > best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': config.MODEL.NAME,
            'state_dict': model.module.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir, filename='checkpoint.pth.tar')

    final_model_state_file = os.path.join(final_output_dir,
                                          'final_state.pth.tar')
    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
