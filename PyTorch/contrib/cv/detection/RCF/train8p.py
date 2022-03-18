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
# limitations under the License.

import torch
import models
import os
from data_loader import BSDS_RCFLoader
from torch.utils.data import DataLoader
import apex.amp as amp
import argparse
from apex.optimizers import NpuFusedSGD
import time

def parse_arg():
    parser = argparse.ArgumentParser(description="train RCF")
    parser.add_argument('--npu', help='npu id', type=str)
    args = parser.parse_args()
    return args.npu

os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29679'

npu = parse_arg()

# 8p
# ----------------------------------------------------------------
torch.distributed.init_process_group(backend='hccl', world_size=8, rank=int(npu))

device = torch.device("npu:{}".format(npu))
torch.npu.set_device(device)
# ----------------------------------------------------------------

model = models.resnet101(pretrained=True).to(device)

# init_lr = 8*1e-2
init_lr = 8*1e-3
batch_size = 3

#resume = 'ckpt/only-final-lr-0.08-iter-40000.pth'
#checkpoint = torch.load(resume)
#model.load_state_dict(checkpoint)

def adjust_lr(init_lr, now_it, total_it):
    power = 0.9
    lr = init_lr * (1 - float(now_it) / total_it) ** power
    return lr

def make_optim(model, lr):
    optim = NpuFusedSGD(params=model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    return optim

def save_ckpt(model, name):
    print('saving checkpoint ... {}'.format(name), flush=True)
    if not os.path.isdir('ckpt'):
        os.mkdir('ckpt')
    torch.save(model.state_dict(), os.path.join('ckpt', '{}.pth'.format(name)))


train_dataset = BSDS_RCFLoader(split="train")
# test_dataset = BSDS_RCFLoader(split="test")
# 添加trainsampler
# ----------------------------------------------------------------
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size,
    num_workers=8, drop_last=True, sampler=train_sampler)
# ----------------------------------------------------------------


def cross_entropy_loss_RCF(prediction, label):
    label = label.long()
    mask = label.float()
    num_positive = torch.sum((mask==1).float()).float()
    num_negative = torch.sum((mask==0).float()).float()

    mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
    mask[mask == 2] = 0

    cost = torch.nn.functional.binary_cross_entropy(
            prediction.float(),label.float(), weight=mask, reduce=False)
    return torch.sum(cost) / (num_negative + num_positive)


model.train()
total_epoch = 30
each_epoch_iter = len(train_loader)
total_iter = total_epoch * each_epoch_iter

print_cnt = 10
ckpt_cnt = 10000
cnt = 0


optim = make_optim(model, adjust_lr(init_lr, cnt, total_iter))
# ---------------------------------------------------------------------------------
model, optim = amp.initialize(model, optim, opt_level="O2",loss_scale=128.0, combine_grad=True)
model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[npu],broadcast_buffers=False,find_unused_parameters=True)
# ---------------------------------------------------------------------------------


for epoch in range(total_epoch):
    avg_loss = 0.
    for i, (image, label) in enumerate(train_loader):
        start = time.time()
        cnt += 1
        
        image, label = image.to(device), label.to(device)
        outs = model(image, label.size()[2:])
        total_loss = cross_entropy_loss_RCF(outs[-1], label)
        optim.zero_grad()
#        total_loss.backward()
        with amp.scale_loss(total_loss, optim) as scaled_loss:
            scaled_loss.backward()
        optim.step()
        fps = 8*batch_size / (time.time() - start)

        avg_loss += float(total_loss)
        if cnt % print_cnt == 0:
            print('[{}/{}] epoch:{} loss:{} avg_loss:{} FPS:{}'.format(cnt, total_iter, epoch, float(total_loss), avg_loss / print_cnt, fps), flush=True)
            avg_loss = 0

        if cnt % ckpt_cnt == 0 and int(npu) % 8 == 0:
            save_ckpt(model.module, 'only-final-lr-{}-iter-{}'.format(init_lr, cnt))

