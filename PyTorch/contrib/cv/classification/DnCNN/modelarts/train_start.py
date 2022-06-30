# -*- coding: utf-8 -*-
# Copyright 2022 Huawei Technologies Co., Ltd
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
''' train DnCNN '''

import os
import sys
import time
import apex
import argparse
import torch
import torch.onnx
import torch.npu
import torch.nn as nn
import moxing as mox
from collections import OrderedDict
from torch.utils.data import DataLoader
from models import DnCNN
from dataset import Traindata, Evldata
from utils import AverageMeter, weights_init_kaiming, batch_PSNR
from apex import amp


sys.setrecursionlimit(300000)
parser = argparse.ArgumentParser(description="mindspore DnCNN training")
parser.add_argument("--preprocess", type=str, default="True", help='write T or Ture means create h5py dataset')
parser.add_argument('--data_url', type=str, required=True, help='path of dataset')
parser.add_argument('--train_url', type=str, required=True, help='where train ckpts saved')
parser.add_argument("--batchSize", type=int, default=512, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=130, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=1.6, help="Initial learning rate")
parser.add_argument("--mode", type=str, default="S", help='with known noise level (S) or blind training (B)')
parser.add_argument("--noiseL", type=float, default=15, help='noise level; ignored when mode=B')
parser.add_argument("--val_noiseL", type=float, default=15, help='noise level used on validation set')
parser.add_argument("--loadWorld", type=int, default=8, help='dataloader worldsize')
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--gpu", default=0, type=int, help='npu id to use')
opt = parser.parse_args()

CACHE_TRAIN_URL = '/cache/train_url'
CACHE_DATA_URL = '/cache/data_url'

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
    CALCULATE_DEVICE = 'npu:{}'.format(opt.gpu)
    torch.npu.set_device(CALCULATE_DEVICE)

    # modelarts: copy from obs
    print('Copying data ...\n')
    if not os.path.exists(CACHE_DATA_URL):
        os.makedirs(CACHE_DATA_URL)
    if not os.path.exists(CACHE_TRAIN_URL):
        os.makedirs(CACHE_TRAIN_URL)
    mox.file.copy_parallel(opt.data_url, CACHE_DATA_URL)

    print('Loading dataset ...\n')
    dataset_train = Traindata(data_path=CACHE_DATA_URL, getDataSet='train')
    dataset_val = Evldata(data_path=CACHE_DATA_URL, getDataSet='Set68')
    loader_train = DataLoader(dataset=dataset_train, num_workers=8, batch_size=512, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))

    net = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
    net.apply(weights_init_kaiming)
    criterion = nn.MSELoss(reduction='sum')
    criterion.to(CALCULATE_DEVICE)
    model = net.to(CALCULATE_DEVICE)
    optimizer = apex.optimizers.NpuFusedAdam(model.parameters(), lr=opt.lr)

    model, optimizer = amp.initialize(model, optimizer, opt_level="O2", loss_scale=128.0)  # apex add

    step_time = AverageMeter('Time', ':6.3f')
    step_start = time.time()
    data_time = AverageMeter('Time', ':6.3f')
    data_start = time.time()

    psnr_val = 0
    maxPsnr = 0
    for epoch in range(opt.epochs):
        current_lr = opt.lr / getDiv(epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)
        model.train()

        for i, loadData in enumerate(loader_train, 0):
            data_difTime = time.time() - data_start
            data_time.update(data_difTime)  # update data load time
            step_start = time.time()

            imgn_train, img_train, noise = loadData

            img_train, imgn_train = img_train.to(CALCULATE_DEVICE), imgn_train.to(CALCULATE_DEVICE)
            noise = noise.to(CALCULATE_DEVICE)

            out_train = model(imgn_train)
            loss = criterion(out_train, noise) / (imgn_train.size()[0] * 2)
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            torch.npu.synchronize()  # npu time sync
            step_difTime = time.time() - step_start
            step_time.update(step_difTime)
            data_start = time.time()

            if i % 210 == 0:
                print('epoch[{}] load[{}/{}] loss: {:.4f} '.format(epoch, i, len(loader_train), loss.item()))
                print('dataTime[{:.3f}]/[{:.3f}]  stepTime[{:.3f}/{:.3f}]  FPS: {:.3f}'.format(
                    data_difTime,
                    data_time.avg,
                    step_difTime,
                    step_time.avg,
                    opt.batchSize / (data_difTime + step_difTime)
                ))

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
            torch.save(model.state_dict(), os.path.join(CACHE_TRAIN_URL, 'net_1p.pth'))  # net602a 31.9

    print("finnal Psnr:", maxPsnr)

def proc_nodes_module(checkpoint):
    ''' proc_nodes_module '''
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        if k[0:7] == "module.":
            name = k[7:]
        else:
            name = k[0:]
        new_state_dict[name] = v
    return new_state_dict

def export():
    ''' export onnx '''
    pretrained_net = torch.load(os.path.join(CACHE_TRAIN_URL, 'net_1p.pth'), map_location='cpu')
    pretrained_net['state_dict'] = proc_nodes_module(pretrained_net)

    model = DnCNN(channels=1, num_of_layers=17)
    model.load_state_dict(pretrained_net['state_dict'])
    model.eval()
    input_names = ["actual_input_1"]
    dummy_input = torch.randn(1, 1, 481, 481)

    dynamic_axes = {'actual_input_1': {0: '-1'}}
    torch.onnx.export(model, dummy_input, os.path.join(CACHE_TRAIN_URL, 'DnCNN.onnx'), dynamic_axes=dynamic_axes, input_names=input_names, opset_version=11)

    mox.file.copy_parallel(CACHE_TRAIN_URL, opt.train_url)

if __name__ == "__main__":
    main()
    export()

