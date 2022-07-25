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
from config import hrnet
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
import moxing as mox

print('os.getcwd():', os.getcwd())

def parse_args():
    parser = argparse.ArgumentParser(description='Train classification network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        type=str,
                        default='/home/work/user-job-dir/code/experiments/cls_hrnet_w44_sgd_lr5e-2_wd1e-4_bs32_x100.yaml')

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
                        default='/cache/training')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='/cache/log')
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
                        default=0)

    parser.add_argument('--bs',
                        help='batch_size',
                        type=int,
                        default=32)

    parser.add_argument('--lr',
                        help='lr',
                        type=float,
                        default=0.1)

    parser.add_argument('--dn',
                        help='device_num',
                        type=str,
                        default=1)
    
    parser.add_argument('--nproc',
                        help='nproc',
                        type=int,
                        default=1)

    parser.add_argument('--train_epochs',
                        help='train epochs',
                        type=int,
                        default=1)

    # 模型输出目录
    parser.add_argument('--train_url',
                        help='the path model saved',
                        type=str,
                        default='')
    # 数据集目录
    parser.add_argument('--data_url',
                        help='the training data',
                        type=str,
                        default='')
    
    args = parser.parse_args()
    update_config(config, args)

    return args


def main():

    args = parse_args()

    # host_ip = args.addr
    device_id = args.device_id
    device_num = int(args.dn)
    
    # os.environ['MASTER_ADDR'] = host_ip
    # os.environ['MASTER_PORT'] = '29988'
    # dist.init_process_group(backend='hccl', rank=rank_id, world_size=device_num)
    
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
    shutil.copytree(os.path.join(this_dir, './lib/models'), models_dst_dir)

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
    model, optimizer = amp.initialize(model, optimizer, opt_level="O2", loss_scale=128.0)
    # model = DDP(model, device_ids=[device_id])

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
    real_path = '/cache/data_url'
    if not os.path.exists(real_path):
        os.makedirs(real_path)
    mox.file.copy_parallel(args.data_url, real_path)
    print("training data finish copy to %s." % real_path)
    
    traindir = os.path.join(real_path, "train")
    valdir = os.path.join(real_path, "val")

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

    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    nproc = args.nproc
    num_workers = nproc // device_num
    bs = args.bs
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=bs*device_num,
        shuffle=(train_dataset is None),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    valid_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(int(config.MODEL.IMAGE_SIZE[0] / 0.875)),
            transforms.CenterCrop(config.MODEL.IMAGE_SIZE[0]),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=bs*device_num,
        shuffle=(train_dataset is None),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    train_epochs = args.train_epochs
    for epoch in range(train_epochs):
        # train_dataset.set_epoch(epoch)
        lr_scheduler.step()
        # train for one epoch
        train(config, train_loader, model, criterion, optimizer, epoch,
              final_output_dir, tb_log_dir, writer_dict, device_num, bs)
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
            'state_dict': model.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir, filename='checkpoint.pth.tar')

    final_model_state_file = os.path.join(final_output_dir,
                                          'final_state.pth.tar')
    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    torch.save(model.state_dict(), final_model_state_file)
    writer_dict['writer'].close()
    
    checkpoint = torch.load(final_model_state_file, map_location='cpu')
    onnx_file_path = os.path.join(final_output_dir, 'hrnet.onnx')
    model = hrnet.get_cls_net(config)
    model.load_state_dict(checkpoint)
    model.eval()
    print(model)

    input_names = ["image_input"]
    output_names = ["output_1"]
    dynamic_axes = {'image': {0: '-1'}, 'class': {0: '-1'}}
    dummy_input = torch.randn(1, 3, 256, 256)
    torch.onnx.export(model, dummy_input, onnx_file_path, input_names=input_names, dynamic_axes=dynamic_axes, output_names=output_names, verbose=True, opset_version=11)

    mox.file.copy_parallel(final_output_dir, args.train_url)
    mox.file.copy_parallel(tb_log_dir, args.train_url)

if __name__ == '__main__':
    main()
