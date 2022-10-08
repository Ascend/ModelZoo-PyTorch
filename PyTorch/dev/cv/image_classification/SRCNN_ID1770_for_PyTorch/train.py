#
# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================
#
import argparse
import os
import copy

import torch
if torch.__version__ >= "1.8":
    import torch_npu
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import time
from models import SRCNN
from datasets import TrainDataset, EvalDataset
from utils import AverageMeter, calc_psnr
import torch.npu
import os

# print("1111111111")
# import torch
# torch.npu.global_step_inc()
# print("2222222222")

# 使能混合精度
try:
    from apex import amp
except ImportError:
    amp = None
import apex
# 使能混合精度

NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')


if __name__ == '__main__':

    # 开启模糊编译
    # print("ttttttttttttttttttt")
    # import torch
    # torch.npu.global_step_inc()
    # print("zzzzzzzzzzzzzzzz")
    # 开启模糊编译

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, required=True)
    parser.add_argument('--eval-file', type=str, required=True)
    parser.add_argument('--outputs-dir', type=str, required=True)
    parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=400)
    parser.add_argument('--num-workers', type=int, default=128)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--ddp', action='store_true', help='default Fasle')

    # 使能混合精度
    parser.add_argument('--apex', action='store_true',
                        help='Use apex for mixed precision training')
    parser.add_argument('--apex_opt_level', default='O1', type=str,
                        help='For apex mixed precision training'
                             'O0 for FP32 training, O1 for mixed precision training.'
                             'For further detail, see https://github.com/NVIDIA/apex/tree/master/examples/imagenet')
    parser.add_argument('--loss_scale_value', default=1024., type=float,
                        help='loss scale using in amp, default -1 means dynamic')
    # 使能混合精度
    args = parser.parse_args()

    if args.ddp:
        NPU_CALCULATE_DEVICE = 0
        if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
            NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
        if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
            torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')
        NPU_WORLD_SIZE = int(os.getenv('NPU_WORLD_SIZE'))
        RANK = int(os.getenv('RANK'))
        torch.distributed.init_process_group('hccl', rank=RANK, world_size=NPU_WORLD_SIZE)

    args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    device = torch.device(f'npu:{NPU_CALCULATE_DEVICE}')

    torch.manual_seed(args.seed)

    model = SRCNN().to(f'npu:{NPU_CALCULATE_DEVICE}')
    criterion = nn.MSELoss()
    #使用PT原生优化器接口
    # optimizer = optim.Adam([
    #     {'params': model.conv1.parameters()},
    #     {'params': model.conv2.parameters()},
    #     {'params': model.conv3.parameters(), 'lr': args.lr * 0.1}
    # ], lr=args.lr)

    #替换亲和性接口，但是不合并paras
    # optimizer = apex.optimizers.NpuFusedAdam([
    #     {'params': model.conv1.parameters()},
    #     {'params': model.conv2.parameters()},
    #     {'params': model.conv3.parameters(), 'lr': args.lr * 0.1}
    # ], lr=args.lr)

    #替换亲和性接口，同时将合并第1和第2组paras
    optimizer = apex.optimizers.NpuFusedAdam([
        {'params': list(model.conv1.parameters()) + list(model.conv2.parameters())},
        {'params': model.conv3.parameters(), 'lr': args.lr * 0.1}
    ], lr=args.lr)




    if args.apex:
        # print("args.apex=============================", args.apex)
        model, optimizer = amp.initialize(model, optimizer,
                                           opt_level=args.apex_opt_level,
                                           loss_scale=args.loss_scale_value,
                                           combine_grad=True)
    #使能混合精度

    if args.ddp:
        model = model.to(f'npu:{NPU_CALCULATE_DEVICE}')
        if not isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[NPU_CALCULATE_DEVICE],
                                                              broadcast_buffers=False)

    train_dataset = TrainDataset(args.train_file)
    if args.ddp:
        train_dataloader_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_dataloader_batch_size = args.batch_size
        train_dataloader = DataLoader(dataset=train_dataset,
                                      batch_size=train_dataloader_batch_size,
                                      shuffle=False,
                                      num_workers=args.num_workers,
                                      pin_memory=True,
                                      drop_last=True, sampler=train_dataloader_sampler)
    else:
        train_dataloader = DataLoader(dataset=train_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=args.num_workers,
                                    pin_memory=True,
                                    drop_last=True)
    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0

    for epoch in range(args.num_epochs):
        if args.ddp:
            train_dataloader.sampler.set_epoch(epoch)
        model.train()
        epoch_losses = AverageMeter()

        # with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size)) as t:
        #     t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

        # iter = 0
        start_time=time.time()
        # print("train_dataloader length============================", len(train_dataloader))
        for i, (inputs, labels) in enumerate(train_dataloader):
            # inputs, labels = data
            inputs = inputs.to(f'npu:{NPU_CALCULATE_DEVICE}', non_blocking=True)
            labels = labels.to(f'npu:{NPU_CALCULATE_DEVICE}', non_blocking=True)

            preds = model(inputs)

            loss = criterion(preds, labels)

            epoch_losses.update(loss.item(), len(inputs))

            optimizer.zero_grad()

            # 使能混合精度
            # loss.backward()
            if args.apex:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            # 使能混合精度

            optimizer.step()

            FPS = args.batch_size/(time.time()-start_time)
            start_time = time.time()
            print("Epoch: {}, iter: {}, FPS: {:.4f}, loss: {:.6f}".format(epoch, i, FPS, epoch_losses.avg))

            # t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
            # t.update(len(inputs))

        torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))

        model.eval()
        epoch_psnr = AverageMeter()

        for data in eval_dataloader:
            inputs, labels = data

            inputs = inputs.to(f'npu:{NPU_CALCULATE_DEVICE}')
            labels = labels.to(f'npu:{NPU_CALCULATE_DEVICE}')

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
