# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
# Copyright 2022 Huawei Technologies Co., Ltd
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
import sys
import archs
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import losses
import moxing as mox

from albumentations.augmentations.functional import optical_distortion
from tqdm import tqdm
from collections import OrderedDict
from torch.optim import lr_scheduler
from torch.utils.data.distributed import DistributedSampler
from dataset import Dataset
from metrics import iou_score
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from utils import AverageMeter, str2bool
from apex import amp
from apex.optimizers import NpuFusedAdam


# ---------modelarts modification-----------------

CACHE_DATA_URL = '/cache/data_url/'
CACHE_TRAINING_URL = '/cache/training'
# ---------modelarts modification end-------------


ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')


def parse_args():
    parser = argparse.ArgumentParser()


    # ------------modelarts modification-----------------
    parser.add_argument('--data_url', metavar='DIR', default='/cache/data_url', help='path to dataset')
    parser.add_argument('--train_url', default="/cache/training", type=str, help="setting dir of training output")
    parser.add_argument('--onnx', default=True, help="convert pth model to onnx")
    # -----------modelarts modification end-------------
    parser.add_argument('--pretrained', default=False, action="store_true")
    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    
    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='UNet',
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
    
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument("--device", default="npu", type=str)
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument("--profile", default=0, type=int)
    config = parser.parse_args()

    return config


def profiling(loader, model, loss_fun, optimizer, loc, config):
    # switch to train mode
    model.train()

    def update(model, images, target, optimizer):
        output = model(images)
        loss = loss_fun(output, target)
        optimizer.zero_grad()
        if config['amp']:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()        
        optimizer.step()

    step = 0
    for images, target, _ in loader:
        if 'npu' in config['device']:
            target = target.to(torch.int32)

        if 'npu' in config['device'] or 'gpu' in config['device']:
            images = images.to(loc, non_blocking=True)
            target = target.to(loc, non_blocking=True)
            
        if step < 5:
            update(model, images, target, optimizer)
        else:
            if config['device'] == 'npu':
                with torch.autograd.profiler.profile(use_npu=True) as prof:
                    update(model, images, target, optimizer)
            elif config['device'] == "gpu":
                with torch.autograd.profiler.profile(use_cuda=True) as prof:
                    update(model, images, target, optimizer)
            break
        step += 1

    prof.export_chrome_trace("output.prof")


def init_process_group(proc_rank, world_size, device_type="npu", port="29588", dist_backend="hccl"):
    """Initializes the default process group."""

    # Initialize the process group
    print("==================================")    
    print('Begin init_process_group')
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = port
    if device_type == "npu":
        torch.distributed.init_process_group(
            backend=dist_backend,
            world_size=world_size,
            rank=proc_rank
        )
    elif device_type == "gpu":
        torch.distributed.init_process_group(
            backend=dist_backend,
            init_method="tcp://{}:{}".format("127.0.0.1", port),
            world_size=world_size,
            rank=proc_rank
        )        

    print("==================================")
    print("Done init_process_group")

    # Set the GPU to use
    #torch.cuda.set_device(proc_rank)
    if device_type == "npu":
        torch.npu.set_device(proc_rank)
    elif device_type == "gpu":
        torch.cuda.set_device(proc_rank)
    print('Done set device', device_type, dist_backend, world_size, proc_rank)


def train(config, train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter('Time', ':6.3f', start_count_index=5)
    data_time = AverageMeter('Data', ':6.3f', start_count_index=5)
    if config['num_gpus'] > 1:
        batch_time = AverageMeter('Time', ':6.3f', start_count_index=3)
        data_time = AverageMeter('Data', ':6.3f', start_count_index=3)
    losses = AverageMeter('Loss', ':6.8f')
    iou = AverageMeter('Iou', ':6.4f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, iou],
        prefix="Epoch: [{}]".format(epoch + 1))

    model.train()

    step = 0
    end = time.time()
    for input, target, _ in train_loader:
        data_time.update(time.time() - end)
        step += 1
        input = input.npu()
        target = target.npu()

        # compute output
        if config['deep_supervision']:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            iou_now = iou_score(outputs[-1], target)
        else:
            output = model(input)
            torch.npu.synchronize()
            loss = criterion(output, target)
            torch.npu.synchronize()
            iou_now = iou_score(output, target)

        losses.update(loss.item(), input.size(0))
        iou.update(iou_now, input.size(0))

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        torch.npu.synchronize()
        if config["amp"]:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
                torch.npu.synchronize()
        else:
            loss.backward()
        optimizer.step()
        torch.npu.synchronize()

        batch_time.update(time.time() - end)
        end = time.time()
        if step == 4 and config['num_gpus'] == 1:
            batch_time.reset()
        if config['num_gpus'] == 1 or (config['num_gpus'] > 1
                                                    and config['rank_id'] % config['num_gpus'] == 0):
            progress.display(step)

    if config['num_gpus'] == 1 or (config['num_gpus'] > 1
                                               and config['rank_id'] % config['num_gpus'] == 0):
        print("[npu id:", config['rank_id'], "]", '* FPS@all {:.3f}'
        .format(config['num_gpus'] * config['batch_size'] / (batch_time.avg * config['num_gpus'])))


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

    with torch.no_grad():
        step = 0
        end = time.time()
        for input, target, _ in val_loader:
            step += 1
            input = input.npu()
            target = target.npu()

            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou_now = iou_score(outputs[-1], target)
            else:
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

    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])
    os.makedirs('models/%s' % config['name'], exist_ok=True)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)


    if not os.path.exists(CACHE_DATA_URL):
        os.makedirs(CACHE_DATA_URL)
    mox.file.copy_parallel(config['data_url'], CACHE_DATA_URL)
    print("training data finish copy to %s." % CACHE_DATA_URL)
    # --------------modelarts modification end--------------

    SEED= 5
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False
    
    if config['num_gpus'] > 1:
        init_process_group(proc_rank=config['rank_id'], world_size=config['num_gpus'], device_type=config['device'])
    elif config['device'] == "npu":
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

    # define loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().npu()
    else:
        criterion = losses.__dict__[config['loss']]().npu()

    cudnn.benchmark = True

    # create model
    if config['pretrained']:
        model = archs.__dict__[config['arch']](config['num_classes'], config['input_channels'])
        checkpoint = torch.load(config['resume'], map_location='cpu')
        model.load_state_dict(checkpoint, strick=False)
    else:
        model = archs.__dict__[config['arch']](config['num_classes'], config['input_channels'])
    model = model.npu()

    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = NpuFusedAdam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config["amp"]:
        model, optimizer = amp.initialize(model, optimizer, opt_level=config["opt_level"], 
                                        loss_scale=config["loss_scale"], combine_grad=True)

    if config["num_gpus"] > 1:
        #Make model replica operate on the current device
        ddp = torch.nn.parallel.DistributedDataParallel
        model = ddp(model, device_ids=[cur_device], broadcast_buffers=False)

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, 
                                            milestones=[int(e) for e in config['milestones'].split(',')], 
                                            gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    # Data loading code
    img_ids = glob(os.path.join(CACHE_DATA_URL, 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

    train_transform = Compose([
        transforms.RandomRotate90(),
        transforms.Flip(),
        OneOf([
            transforms.HueSaturationValue(),
            transforms.RandomBrightness(),
            transforms.RandomContrast(),
        ], p=1),
        transforms.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_transform = Compose([
        transforms.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join(CACHE_DATA_URL, 'images'),
        mask_dir=os.path.join(CACHE_DATA_URL, 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=train_transform)
    
    train_sampler = DistributedSampler(train_dataset) if config['num_gpus'] > 1 else None

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(CACHE_DATA_URL, 'images'),
        mask_dir=os.path.join(CACHE_DATA_URL, 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)

    val_sampler = DistributedSampler(val_dataset) if config['num_gpus'] > 1 else None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=int(config['batch_size']/ config['num_gpus']),
        shuffle=(False if train_sampler else True),
        sampler=train_sampler,
        num_workers=int(config['num_workers'] / config['num_gpus']),
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=int(config['batch_size'] / config['num_gpus']),
        shuffle=False,
        #sampler=val_sampler,
        num_workers=int(config['num_workers'] / config['num_gpus']),
        drop_last=False)

    if config['evaluate']:
        val_iou = validate(config, val_loader, model, criterion)
        return

    if config['profile']:
        profiling(train_loader, model, criterion, optimizer, loc, config)
        return

    best_iou = 0
    trigger = 0
    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch+1, config['epochs']))
        if config['num_gpus'] > 1:
            train_sampler.set_epoch(epoch)
        # train for one epoch
        train(config, train_loader, model, criterion, optimizer, epoch)
        # evaluate on validation set
        val_iou = validate(config, val_loader, model, criterion)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()

        is_best = val_iou > best_iou
        best_iou = max(val_iou, best_iou)
        trigger += 1

        if is_best:
            trigger = 0

        # save checkpoint
        if config['num_gpus'] == 1 or \
                (config['num_gpus'] > 1 and config['rank_id'] % config['num_gpus'] == 0):
            print("save epoch is ", epoch + 1)
            if config['amp']:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_iou': best_iou,
                    'optimizer': optimizer.state_dict(),
                    'amp': amp.state_dict(),
                }, is_best, epoch, config['name'], config)
            else:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_iou': best_iou,
                    'optimizer': optimizer.state_dict(),
                }, is_best, epoch, config['name'], config)      

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break
        
        torch.cuda.empty_cache()
    mox.file.copy_parallel(CACHE_TRAINING_URL, config['train_url'])


def save_checkpoint(state, is_best, epoch, cfg_name, config):
    directory = '/cache/training'
    if not os.path.exists(directory):
        os.makedirs(directory)
    c_filename = directory + "/" + "checkpoint.pth.tar"
    b_filename = directory + "/" + "model_best.pth.tar"
    torch.save(state, c_filename)
    if is_best:
        print("======save best", " epoch ", epoch, "=======")
        mox.file.copy(c_filename, b_filename)
        onnx_file = os.path.join(directory, 'UNet.onnx')
        if config["amp"] and config["opt_level"] == 'O1':
            from apex import amp
            with amp.disable_casts():
                convert(b_filename, onnx_file)
        else:
            convert(b_filename, onnx_file)         


def proc_node_module(checkpoint, AttrName):
    new_state_dict = OrderedDict()
    for k, v in checkpoint[AttrName].items():
        if(k[0:7] == "module."):
            name = k[7:]
        else:
            name = k[0:]
        new_state_dict[name] = v
    return new_state_dict


def convert(pth_file_path, onnx_file_path):
    checkpoint = torch.load(pth_file_path, map_location='cpu')
    checkpoint['state_dict'] = proc_node_module(checkpoint, 'state_dict')
    model = archs.__dict__["UNet"](1, 3)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print(model)
    
    input_names = ["actual_input_1"]
    output_names = ["output1"]
    dummy_input = torch.randn(16, 3, 224, 224)
    torch.onnx.export(model, dummy_input, onnx_file_path, input_names=input_names, output_names=output_names,
                      opset_version=11)


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
