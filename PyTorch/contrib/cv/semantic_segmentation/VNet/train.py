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

import time
import argparse
import torch

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.multiprocessing import spawn
import torchvision.transforms as transforms
import torchbiomed.datasets as dset

import os
import shutil

import vnet
from apex import amp
import apex

nodule_masks = "normalized_nodule_mask"
lung_masks = "normalized_lung_mask"
ct_images = "normalized_lung_ct"
ct_targets = lung_masks
target_split = [2, 2, 2]
best_prec = 100.

os.environ['MASTER_ADDR'] = '127.0.0.1' 
os.environ['MASTER_PORT'] = '333' 

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.zero_()

def datestr():
    now = time.gmtime()
    return '{}{:02}{:02}_{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)


def save_checkpoint(state, is_best, path, prefix, filename='checkpoint.pth.tar'):
    prefix_save = os.path.join(path, prefix)
    name = prefix_save + '_' + filename
    torch.save(state, name)
    if is_best:
        shutil.copyfile(name, prefix_save + '_model_best.pth.tar')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=10)
    parser.add_argument('--device', type=str, default='gpu')
    parser.add_argument('--device_num', type=int, default=1)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--nEpochs', type=int, default=200)
    parser.add_argument('--lr',type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=float, default=1)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--data', type=str, default='/opt/npu/dataset/luna16')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    # 1e-8 works well for lung masks but seems to prevent
    # rapid learning for nodule masks
    parser.add_argument('--weight_decay', '--wd', default=1e-8, type=float,
                        metavar='W', help='weight decay (default: 1e-8)')
    parser.add_argument('--save', help='save path')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--loss_scale', type=int, default=128)
    parser.add_argument('--opt_level',type=str, default='O2')
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--dist_backend', type=str, default='nccl',
                        help='use nccl or hccl')
    parser.add_argument('--dist_url', type=str)
    parser.add_argument('--world_size', type=int, default=1,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--print', type=bool, default=False)
    args = parser.parse_args()
    args.save = args.save or 'work/vnet.base.{}'.format(datestr())

    torch.manual_seed(args.seed)
    if args.device=='npu':
        torch.npu.manual_seed(args.seed)
    else:
        torch.cuda.manual_seed(args.seed)

    ngpus_per_node = args.device_num
    if args.distributed:
        args.world_size = ngpus_per_node * args.world_size
        spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.device_id, ngpus_per_node, args)



def main_worker(gpu, ngpus_per_node, args):
  
    global best_prec
    if args.device=='npu':
        args.device_idx = 'npu:{}'.format(gpu)
    else:
        args.device_idx = 'cuda:{}'.format(gpu)
    batch_size = args.batchSz
    if gpu + 1 == ngpus_per_node:
        args.print = True

    if args.print:
        print("build vnet")
    model = vnet.VNet(elu=False, nll=True)

    args.start_epoch = 1
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location=args.device_idx)
            args.start_epoch = checkpoint['epoch']
            best_prec = checkpoint['error']
            model.load_state_dict(checkpoint['state_dict'])
            if args.print:
                print("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        model.apply(weights_init)

    if args.print:
        if os.path.exists(args.save):
            shutil.rmtree(args.save)
        os.makedirs(args.save, exist_ok=True)
    
    # LUNA16 dataset isotropically scaled to 2.5mm^3
    # and then truncated or zero-padded to 160x128x160
    normMu = [-642.794]
    normSigma = [459.512]
    normTransform = transforms.Normalize(normMu, normSigma)
    trainTransform = transforms.Compose([
        transforms.ToTensor(),
        normTransform
    ])
    testTransform = transforms.Compose([
        transforms.ToTensor(),
        normTransform
    ])

    if args.device=='npu':
        optimizer = apex.optimizers.NpuFusedAdam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,  milestones = [50,100], gamma = args.lr_decay, last_epoch=-1)

    if args.distributed:
        args.rank = gpu
        if args.device == 'npu':
            torch.npu.set_device(args.device_idx)
            torch.distributed.init_process_group(backend=args.dist_backend, # init_method="env://", 
                                            world_size=args.world_size, rank=args.rank)
        else:
            torch.cuda.set_device(args.device_idx)
            torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                        world_size=args.world_size, rank=args.rank)

    model = model.to(args.device_idx)
    if args.amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level, loss_scale=args.loss_scale)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.rank], broadcast_buffers=False)

    trainSet = dset.LUNA16(root=args.data, images=ct_images, targets=ct_targets,
                           mode="train", transform=trainTransform, split=target_split, masks=None)
    testSet = dset.LUNA16(root=args.data, images=ct_images, targets=ct_targets,
                    mode="test", transform=testTransform, masks=None, split=target_split)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainSet)
        test_sampler = None
    else:
        train_sampler=None
        test_sampler=None
    trainLoader = DataLoader(trainSet, batch_size=batch_size, shuffle=(train_sampler is None), num_workers=args.workers, sampler=train_sampler)
    testLoader = DataLoader(testSet, batch_size=batch_size, shuffle=False, num_workers=args.workers, sampler=test_sampler)

    target_mean = trainSet.target_mean()
    bg_weight = target_mean / (1. + target_mean)
    fg_weight = 1. - bg_weight
    class_weights = torch.FloatTensor([bg_weight, fg_weight])
    class_weights = class_weights.to(args.device_idx)

    trainF = open(os.path.join(args.save, 'train.csv'), 'w')
    testF = open(os.path.join(args.save, 'test.csv'), 'w')
    for epoch in range(args.start_epoch, args.nEpochs + 1):
        if train_sampler: train_sampler.set_epoch(epoch)
        if test_sampler: test_sampler.set_epoch(epoch)
        if scheduler: scheduler.step()
        train(args, epoch, model, trainLoader, optimizer, trainF, class_weights)
        err = test(args, epoch, model, testLoader, testF, class_weights)
        is_best = False
        if err < best_prec:
            is_best = True
            best_prec = err
            
        if args.print:
            if args.distributed:
                state = model.module.state_dict()
            else:
                state = model.state_dict()
            save_checkpoint({'epoch': epoch,
                            'state_dict': state,
                            'error': best_prec},
                            is_best, args.save, "vnet")

    trainF.close()
    testF.close()


def train(args, epoch, model, trainLoader, optimizer, trainF, weights):
    model.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset) // args.device_num
    if len(trainLoader.dataset) % args.device_num:
        nTrain += 1
    for batch_idx, (data, target) in enumerate(trainLoader):
        start_time=time.time()
        data, target = data.to(args.device_idx), target.to(args.device_idx)
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        output = output.view(-1,2)
        target = target.view(target.numel())
        loss = F.nll_loss(output, target, weight=weights)
        if np.isnan(loss.data.item()):
            print('Get NaN')
            raise AssertionError
        if args.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        nProcessed += len(data)
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        incorrect = pred.ne(target.data).cpu().sum()
        err = 100.*incorrect/target.numel()
        partialEpoch = epoch + batch_idx / len(trainLoader) - 1
        if args.print:
            fps = 1 / (time.time()-start_time) * data.size(0) * args.device_num
            print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)] Loss: {:.4f} Error: {:.4f}% Lr: {:.4f} FPS: {:.4f}'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
            loss.item(), err, optimizer.state_dict()['param_groups'][0]['lr'], fps))
            trainF.write('{:.2f},{:.4f},{:.4f},{:.4f}\n'.format(partialEpoch, loss.item(), err, fps))
            trainF.flush()


def test(args, epoch, model, testLoader, testF, weights):
    model.eval()
    test_loss = 0
    incorrect = 0
    numel = 0
    with torch.no_grad():
        for data, target in testLoader:
            data, target = data.to(args.device_idx), target.to(args.device_idx)
            target = target.view(target.numel())
            numel += target.numel()
            output = model(data)
            output = output.view(-1,2)
            test_loss += F.nll_loss(output, target, weight=weights).item()
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            incorrect += pred.ne(target.data).cpu().sum().item()

    test_loss /= len(testLoader)  # loss function already averages over batch size
    err = 100.*incorrect/numel
    if args.print:
        print('\nTest set: Average loss: {:.4f}, Error: {}/{} ({:.4f}%)\n'.format(
        test_loss, incorrect, numel, err))
        testF.write('{},{:.4f},{:.4f}\n'.format(epoch, test_loss, err))
        testF.flush()
    return err

if __name__ == '__main__':
    main()
