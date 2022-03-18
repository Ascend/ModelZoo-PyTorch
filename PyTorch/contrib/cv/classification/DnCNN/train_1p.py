# -*- coding: utf-8 -*- 
# Copyright 2020 Huawei Technologies Co., Ltd
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
import os
import argparse
import numpy as np
import torch
import torch.npu
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models import DnCNN
from dataset import Traindata, Evldata
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import AverageMeter, weights_init_kaiming, batch_PSNR
import time
import apex
from apex import amp

import sys

sys.setrecursionlimit(300000)
parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("--preprocess", type=str, default="False", help='write T or Ture means create h5py dataset')
parser.add_argument('--data_path', type=str, help='path of dataset')
parser.add_argument("--batchSize", type=int, default=512, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--outf", type=str, default=".", help='path of log files')
parser.add_argument("--mode", type=str, default="S", help='with known noise level (S) or blind training (B)')
parser.add_argument("--noiseL", type=float, default=25, help='noise level; ignored when mode=B')
parser.add_argument("--val_noiseL", type=float, default=25, help='noise level used on validation set')
parser.add_argument("--loadWorld", type=int, default=8, help='dataloader worldsize')
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--gpu", default=0, type=int, help='npu id to use')
opt = parser.parse_args()


def getDiv(epo):
    """ lr div value get """
    if epo < 5:
        return 1.
    if epo < 10:
        return 2.
    if epo < 15:
        return 3.
    if epo < 20:
        return 4.
    if epo < 30:
        return 4.2
    if epo < 40:
        return 4.6
    if epo < 60:
        return 4.8
    return 1.


def main():
    """ main fun """
    # 指定训练卡  
    CALCULATE_DEVICE = 'npu:{}'.format(opt.gpu)
    torch.npu.set_device(CALCULATE_DEVICE)

    # 数据准备
    print('Loading dataset ...\n')
    dataset_train = Traindata(data_path=opt.data_path, getDataSet='train')
    dataset_val = Evldata(data_path=opt.data_path, getDataSet='Set68')
    loader_train = DataLoader(dataset=dataset_train, num_workers=8, batch_size=512, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))

    # 搭建网络    
    net = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
    net.apply(weights_init_kaiming)
    criterion = nn.MSELoss(reduction='sum')
    criterion.to(CALCULATE_DEVICE)
    model = net.to(CALCULATE_DEVICE)
    # 优化函数搭建
    optimizer = apex.optimizers.NpuFusedAdam(model.parameters(), lr=opt.lr)

    model, optimizer = amp.initialize(model, optimizer, opt_level="O2", loss_scale=128.0)  # apex add

    # # 多进程初始化,初始化通信环境
    # os.environ['MASTER_ADDR'] = '127.0.0.1'
    # os.environ['MASTER_PORT'] = '29688'
    # os.environ['WORLD_SIZE'] = '1'
    # dist.init_process_group(backend="hccl", init_method='env://', world_size=-1, rank=0)
    # model = DDP(net, device_ids=[0]).to(CALCULATE_DEVICE)

    # 时间记录申明
    step_time = AverageMeter('Time', ':6.3f')
    step_start = time.time()
    data_time = AverageMeter('Time', ':6.3f')
    data_start = time.time()

    psnr_val = 0
    maxPsnr = 0
    # 开始训练
    for epoch in range(opt.epochs):
        current_lr = opt.lr / getDiv(epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)
        model.train()

        for i, loadData in enumerate(loader_train, 0):
            # 单步的训练
            data_difTime = time.time() - data_start
            data_time.update(data_difTime)  # 数据加载时间更新
            step_start = time.time()

            imgn_train, img_train, noise = loadData

            img_train, imgn_train = img_train.to(CALCULATE_DEVICE), imgn_train.to(CALCULATE_DEVICE)
            noise = noise.to(CALCULATE_DEVICE)

            # print(imgn_train.shape)
            out_train = model(imgn_train)
            loss = criterion(out_train, noise) / (imgn_train.size()[0] * 2)
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            torch.npu.synchronize()  # npu时间同步
            step_difTime = time.time() - step_start
            step_time.update(step_difTime)
            data_start = time.time()  # data的时间

            if i % 210 == 0:
                print('epoch[{}] load[{}/{}] loss: {:.4f} '.format(epoch, i, len(loader_train), loss.item()))
                print('dataTime[{:.3f}]/[{:.3f}]  stepTime[{:.3f}/{:.3f}]  FPS: {:.3f}'.format(data_difTime, \
                                                                                               data_time.avg,
                                                                                               step_difTime,
                                                                                               step_time.avg,
                                                                                               opt.batchSize / (
                                                                                                           data_difTime + step_difTime)))

        ## 一批次训练结束，固定模型推理
        print("------eval model  valData len is ------- ", len(dataset_val))
        model.eval()
        psnr_val = 0
        for k in range(len(dataset_val)):
            imgn_val, img_val, noise = dataset_val[k]
            img_val = torch.unsqueeze(img_val, 0)
            imgn_val = torch.unsqueeze(imgn_val, 0)

            img_val, imgn_val = img_val.to(CALCULATE_DEVICE), imgn_val.to(CALCULATE_DEVICE)

            out_val = torch.clamp(imgn_val - model(imgn_val), 0., 1.)
            psnr_val += batch_PSNR(out_val, img_val, 1.)

        psnr_val /= len(dataset_val)

        print("\n[epoch %d] PSNR_val: %.4f" % (epoch, psnr_val))
        if psnr_val > maxPsnr:
            maxPsnr = psnr_val
            torch.save(model.state_dict(), os.path.join(opt.outf, 'net_1p.pth'))  # net602a 31.9

    print("finnal Psnr:", maxPsnr)


if __name__ == "__main__":
    main()
