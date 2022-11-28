# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ============================================================================
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
# limitations under the License.
import argparse
import os
import random
import shutil
import time
import warnings
from glob import glob
from albumentations.augmentations.functional import optical_distortion
from tqdm import tqdm
from collections import OrderedDict
import numpy as np

import cv2
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
from torch.utils.data.distributed import DistributedSampler

import archs
import losses
from dataset import Dataset
from metrics import iou_score
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from utils import AverageMeter, str2bool
from apex import amp

ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default="UNetpp_Demo",
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    
    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='NestedUNet',
                        choices=ARCH_NAMES,
                        help='model architecture: ' +
                        ' | '.join(ARCH_NAMES) +
                        ' (default: NestedUNet)')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=96, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=96, type=int,
                        help='image height')
    
    # loss
    parser.add_argument('--loss', default='LovaszHingeLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                        ' | '.join(LOSS_NAMES) +
                        ' (default: BCEDiceLoss)')
    
    # dataset
    parser.add_argument('--dataset', default='dsb2018_96',
                        help='dataset name')
    parser.add_argument('--data_path', default='./inputs/dsb2018_96',
                        help='data dir')
    parser.add_argument('--img_ext', default='.png',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.png',
                        help='mask file extension')

    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')

    # amp
    parser.add_argument('--amp', default=True, action='store_true', 
                        help='use amp to train the model')
    parser.add_argument('--loss-scale', default=128., type=float,
                        help='loss scale using in amp, default -1 means dynamic')
    parser.add_argument('--opt-level', default='O2', type=str,
                        help='loss scale using in amp, default -1 means dynamic')

    # dist
    parser.add_argument("--rank_id", dest="rank_id", default=0, type=int)
    parser.add_argument("--num_gpus", default=1, type=int)
    parser.add_argument("--addr", default="127.0.0.1", type=str)
    parser.add_argument("--port", default="29588", type=str)
    parser.add_argument("--dist_backend", default="hccl", type=str) 
    
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument("--device", default="npu", type=str)
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--resume', default="./models/dsb2018_96_NestedUNet_woDS/model_best.pth.tar", type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    config = parser.parse_args()

    return config


def proc_node_module(checkpoint, attr_name):
    new_state_dict = OrderedDict()
    for k, v in checkpoint[attr_name].items():
        if(k[0: 7] == "module."):
            name = k[7:]
        else:
            name = k[0:]
        new_state_dict[name] = v
    return new_state_dict


def validate(config, val_loader, model, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    iou = AverageMeter('Iou', ':6.4f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, iou],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)

    with torch.no_grad():
        step = 0
        end = time.time()
        for input, target, meta in val_loader:
            step += 1
            input = input.npu()
            target = target.npu()

            # compute output
            output = model(input)
            torch.npu.synchronize()
            loss = criterion(output, target)
            torch.npu.synchronize()
            iou_now = iou_score(output, target)

            losses.update(loss.item(), input.size(0))
            iou.update(iou_now, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            output = torch.sigmoid(output).cpu().numpy()
            for i in range(len(output)):
                for c in range(config['num_classes']):
                    cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.jpg'),
                                (output[i, c] * 255).astype('uint8'))
            if config['num_gpus'] == 1 or (config['num_gpus'] > 1
                                                        and config['rank_id'] % config['num_gpus'] == 0):
                progress.display(step)
        if config['num_gpus'] == 1 or \
                (config['num_gpus'] > 1 and config['rank_id'] % config['num_gpus'] == 0):
            print("[npu id:", config['rank_id'], "]", '[AVG-IOU] * Iou {iou.avg:.4f}'
                  .format(iou=iou))

    return iou.avg


def main():
    config = vars(parse_args())
    
    if config['device'] == "npu":
        torch.npu.set_device(0)
    elif config['device'] == "gpu":
        torch.cuda.set_device(0)

    loc = ""
    if config['device'] == "npu":  
        cur_device = torch.npu.current_device()
        loc = "npu:" + str(cur_device)
    elif config['device'] == "gpu":
        cur_device = torch.cuda.current_device()
        loc = "cuda:" + str(cur_device)
    print('cur_device: ', cur_device)

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'])
    model = model.npu()

    # define loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().npu()
    else:
        criterion = losses.__dict__[config['loss']]().npu()

    # optionally resume from a checkpoint
    if config['resume']:
        if os.path.isfile(config['resume']):
            print("=> loading checkpoint '{}'".format(config['resume']))
            checkpoint = torch.load(config['resume'], map_location=loc)
            checkpoint["state_dict"] = proc_node_module(checkpoint, "state_dict")
            best_iou = checkpoint['best_iou']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(config['resume'], checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(config['resume']))

    # Data loading code
    img_ids = glob(os.path.join(config['data_path'], 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

    val_transform = Compose([
        transforms.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(config['data_path'], 'images'),
        mask_dir=os.path.join(config['data_path'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=int(config['batch_size'] / config['num_gpus']),
        shuffle=False,
        num_workers=int(config['num_workers'] / config['num_gpus']),
        drop_last=False)
  
    val_iou = validate(config, val_loader, model, criterion)
        
    torch.cuda.empty_cache()


class AverageMeter(object):
    """Computes and stores the average and current value""" 
    def __init__(self, name, fmt=':f', start_count_index=0):
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
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == '__main__':
    main()
