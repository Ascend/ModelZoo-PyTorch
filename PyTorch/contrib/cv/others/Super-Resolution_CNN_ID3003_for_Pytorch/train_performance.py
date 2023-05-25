#!/usr/bin/env python
# coding=utf-8

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
# ============================================================================

import argparse
import os
import copy

import torch
if torch.__version__ >= '1.8':
    import torch_npu
import torch.npu
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

import apex
from apex import amp

from model import SRCNN
from datasets import TrainDataset, EvalDataset
from utils import AverageMeter, calc_psnr
import time

CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != CALCULATE_DEVICE:
    CALCULATE_DEVICE = f'npu:{CALCULATE_DEVICE}'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_url',type=str,
                        default='./inputs/data_url_0/',
                        )
    parser.add_argument('--train_url', type=str,
                        default='./outputs/train_url_0/',
                        )
    parser.add_argument('--outputs-dir', type=str, default='output',required=False)
    parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--num-epochs', type=int, default=500)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    device = torch.npu.set_device(CALCULATE_DEVICE)

    torch.manual_seed(args.seed)

    # model = SRCNN().to(device)
    model = SRCNN().npu()
    criterion = nn.MSELoss().npu()
    optimizer = apex.optimizers.NpuFusedAdam([
        {'params': model.conv1.parameters()},
        {'params': model.conv2.parameters()},
        {'params': model.conv3.parameters(), 'lr': args.lr*0.1}
    ],lr=args.lr)

    model, optimizer = amp.initialize(model,optimizer,opt_level='O2',loss_scale=32.0,combine_grad=True)
    train_dataset = TrainDataset(os.path.join(args.data_url,'train','T91_x3.h5'))
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True)

    eval_dataset = EvalDataset(os.path.join(args.data_url,'val','Set5_x3.h5'))
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0

    for epoch in range(args.num_epochs):
        model.train()
        epoch_losses = AverageMeter()
        batch_time = AverageMeter()

        i = 0
        runtime = 0
        train_loss = 0.0

        for data in train_dataloader:
            start = time.time()
            inputs, labels = data

            inputs = inputs.npu()
            labels = labels.npu()
            preds = model(inputs)
            preds = torch_npu.npu_format_cast(preds, 2)
            loss = criterion(preds, labels)


            optimizer.zero_grad()
            # loss.backward()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
            train_loss += loss.item()
            runtime += (time.time() - start)
            i+=1
            if i < 3 and epoch == 0:
                print("Step Time: ", time.time() - start)
        print('epoch{}'.format(epoch) + ' : ' + 'loss={:.6f}, each_step_time={:.6f}'.format(train_loss/i, runtime/i))
        torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))

        model.eval()
        epoch_psnr = AverageMeter()

        for data in eval_dataloader:
            inputs, labels = data

            inputs = inputs.npu()
            labels = labels.npu()

            with torch.no_grad():
                preds = model(inputs).clamp(0.0, 1.0)

            epoch_psnr.update(calc_psnr(preds, labels), len(inputs))

        print('eval psnr: {:.2f}'.format(epoch_psnr.avg))

        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())

    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
    torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))
