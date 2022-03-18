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
#from __future__ import print_function
import datetime
import os
import time
import sys

import torch
import torch.utils.data
from torch import nn
import random
import numpy as np
import utils
import argparse
try:
    import apex
    from apex import amp
except ImportError:
    amp = None
from evaluation import *
from data_loader import get_dist_loader, get_loader
from network import R2AttU_Net


def train_one_epoch(model_unet, criterion, optimizer, data_loader, device, epoch, config):
    model_unet.train()
    metric_logger = utils.MetricLogger(delimiter="  ")

    epoch_loss = 0. 
    acc = 0.	# Accuracy
    SE = 0.		# Sensitivity (Recall)
    SP = 0.		# Specificity
    PC = 0. 	# Precision
    F1 = 0.		# F1 Score
    JS = 0.		# Jaccard Similarity
    DC = 0.		# Dice Coefficient
    length = 0
    threshold = 0.5
    steps = len(data_loader)
    for i, (images, GT) in enumerate(data_loader):
        # GT : Ground Truth
        images = images.to(device)
        GT = GT.to(device)
        if i == 5:
            start_time = time.time()

        # SR : Segmentation Result
        SR = model_unet(images)
        SR_probs = torch.nn.functional.sigmoid(SR)
        SR_flat = SR_probs.view(SR_probs.size(0),-1)

        GT_flat = GT.view(GT.size(0),-1)
        loss = criterion(SR_flat,GT_flat)
        epoch_loss += loss.item()

        # Backprop + optimize
        model_unet.zero_grad()
        if config.use_apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        SR_ac = SR > threshold
        GT_ac = GT == torch.max(GT)
        acc += get_accuracy(SR_ac, GT_ac)
        SE += get_sensitivity(SR_ac, GT_ac)
        SP += get_specificity(SR_ac, GT_ac)
        PC += get_precision(SR_ac, GT_ac)
        F1 += get_F1(SR_ac, GT_ac)
        JS += get_JS(SR_ac, GT_ac)
        DC += get_DC(SR_ac, GT_ac)
        length += 1
    acc = acc/length
    
    batch_size = config.batch_size
    fps = batch_size *(steps-5) / (time.time() - start_time)
    metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
    metric_logger.meters['acc'].update(acc, n=batch_size)
    metric_logger.meters['img/s'].update(fps)
    print("Traing, Epoch: %d, Loss: %.4f"%(epoch, loss.item()))

    return acc, fps


def evaluate(model_unet, criterion, valid_loader, device):
    metric_logger = utils.MetricLogger(delimiter="  ")
    model_unet.eval()
    threshold = 0.5
    acc = 0.	# Accuracy
    SE = 0.		# Sensitivity (Recall)
    SP = 0.		# Specificity
    PC = 0. 	# Precision
    F1 = 0.		# F1 Score
    JS = 0.		# Jaccard Similarity
    DC = 0.		# Dice Coefficient
    length=0
    for i, (images, GT) in enumerate(valid_loader):

        images = images.to(device)
        GT = GT.to(device)
        SR = torch.nn.functional.sigmoid(model_unet(images))
        SR_ac = SR > threshold
        GT_ac = GT == torch.max(GT)
        acc += get_accuracy(SR_ac, GT_ac)
        SE += get_sensitivity(SR_ac, GT_ac)
        SP += get_specificity(SR_ac, GT_ac)
        PC += get_precision(SR_ac, GT_ac)
        F1 += get_F1(SR_ac, GT_ac)
        JS += get_JS(SR_ac, GT_ac)
        DC += get_DC(SR_ac, GT_ac)
        metric_logger.synchronize_between_processes(device)
            
        length += 1
        
    acc = acc/length
    SE = SE/length
    SP = SP/length
    PC = PC/length
    F1 = F1/length
    JS = JS/length
    DC = DC/length
    unet_score = acc#JS + DC
    batch_size = images.shape[0]
    metric_logger.meters['acc'].update(acc, n=batch_size)
    return acc

def init_distributed_mode(args):
    if 'RANK_SIZE' in os.environ and 'RANK_ID' in os.environ:
        args.rank_size = int(os.environ['RANK_SIZE'])
        args.rank_id = int(os.environ['RANK_ID'])
        args.device_id = args.rank_id
        args.batch_size = int(args.batch_size / args.rank_size)
        args.num_workers = int((args.num_workers) / args.rank_size)
    else:
        raise RuntimeError("init_distributed_mode failed.")

    torch.distributed.init_process_group(backend='hccl',
                                         world_size=args.rank_size, rank=args.rank_id)
def main(config):
    #设置环境变量
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29688'

    #设置seed
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    os.environ['PYTHONHASHSEED'] = str(1234)

    if config.use_apex:
        if sys.version_info < (3, 0):
            raise RuntimeError("Apex currently only supports Python 3. Aborting.")
        if amp is None:
            raise RuntimeError("Failed to import apex. Please install apex from https://www.github.com/nvidia/apex "
                               "to enable mixed-precision training.")

    # Create directories if not exist
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)

    if config.distributed:
        init_distributed_mode(config)

    config.is_master_node = not config.distributed or config.device_id == 0
    if config.is_master_node:
        print(config)

    device = torch.device(f'npu:'+str(config.device_id))
    torch.npu.set_device(device)

    # Data loading code
    print("Loading data")
    config.train_path = os.path.join(config.data_path, "train")
    config.valid_path = os.path.join(config.data_path, "valid")
    print("Creating data loaders")
    if config.distributed:
        train_loader = get_dist_loader(image_path=config.train_path,
                                image_size=config.image_size,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers,
                                mode='train',
                                augmentation_prob=config.augmentation_prob)
        valid_loader = get_loader(image_path=config.valid_path,
                                image_size=config.image_size,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers,
                                mode='valid',
                                augmentation_prob=0.)
    else:
        train_loader = get_loader(image_path=config.train_path,
                                image_size=config.image_size,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers,
                                mode='train',
                                augmentation_prob=config.augmentation_prob)
        valid_loader = get_loader(image_path=config.valid_path,
                                image_size=config.image_size,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers,
                                mode='valid',
                                augmentation_prob=0.)
    model_unet = R2AttU_Net(img_ch=3, output_ch=1,t=config.t)
    model_unet = model_unet.to(device)

    criterion = torch.nn.BCELoss()
    optimizer = apex.optimizers.NpuFusedAdam(list(model_unet.parameters()), 
        config.lr, [config.beta1, config.beta2])
    if config.use_apex:
        model_unet, optimizer = amp.initialize(model_unet, optimizer, 
                                    opt_level=config.apex_level,loss_scale=config.loss_scale, combine_grad=True)

    model_without_ddp = model_unet
    if config.distributed:
        model_unet = torch.nn.parallel.DistributedDataParallel(model_unet, device_ids=[config.device_id])
        model_without_ddp = model_unet.module

    if config.is_master_node:
        print("Start training")
    start_time = time.time()
    best_unet_score = 0.
    lr = config.lr
    for epoch in range(config.num_epochs):
        acc, fps = train_one_epoch(model_unet, criterion, optimizer, train_loader, device, epoch, config)

        unet_score = evaluate(model_unet, criterion, valid_loader, device=device)
        if config.is_master_node:
            print("Traing, Epoch: %d, Avgacc: %.3f, FPS: %.2f"%(epoch, acc, fps))
            print('Test, Acc: %.3f'%(unet_score))
        if config.is_master_node and config.result_path:
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': config}
            utils.save_on_master(
                checkpoint,
                os.path.join(config.result_path, 'model_{}.pth'.format(epoch)))
            if unet_score > best_unet_score:
                best_unet_score = unet_score
                utils.save_on_master(
                    checkpoint,
                    os.path.join(config.result_path, 'checkpoint.pth'))
        if (epoch+1) % 10 == 0:
            lr = lr/2.
            # lr -= (config.lr / float(config.num_epochs_decay))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print ('Decay learning rate to lr: {}.'.format(lr))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if config.is_master_node:
        print('Training time {}'.format(total_time_str))
    exit()


def parse_args():
    parser = argparse.ArgumentParser()

    
    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--t', type=int, default=3, help='t for Recurrent step of R2U_Net or R2AttU_Net')
    
    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=3)
    parser.add_argument('--output_ch', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_epochs_decay', type=int, default=70)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)        # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)      # momentum2 in Adam    
    parser.add_argument('--augmentation_prob', type=float, default=0.4)

    parser.add_argument('--log_step', type=int, default=2)
    parser.add_argument('--val_step', type=int, default=2)

    # misc
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_type', type=str, default='U_Net', help='U_Net/R2U_Net/AttU_Net/R2AttU_Net')
    parser.add_argument('--test_model_path', type=str, default='./models')
    parser.add_argument('--data_path', type=str, default='./dataset/train/')
    parser.add_argument('--result_path', type=str, default='./result_1p')

    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--use_apex', type=int, default=1)
    parser.add_argument('--apex_level', type=str, default="O2")
    parser.add_argument('--loss_scale', type=float, default=128.)

    parser.add_argument('--world_size', type=int, default=8)
    parser.add_argument('--distributed', type=int, default=0,
                        help='Use multi-processing distributed training to launch.')

    config = parser.parse_args()
    main(config)

if __name__ == "__main__":
    args = parse_args()
    main(args)
