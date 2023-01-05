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
#from __future__ import print_function, division
import torch
if torch.__version__ >= "1.8":
    import torch_npu
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision
from torchvision import transforms, datasets, models
import os
import cv2
import time
import argparse
# from model.residual_attention_network_pre import ResidualAttentionModel
# based https://github.com/liudaizong/Residual-Attention-Network
from model.residual_attention_network import ResidualAttentionModel_92_32input_update as ResidualAttentionModel

import apex
from apex import amp

model_file = 'model_92_sgd.pkl'

parser = argparse.ArgumentParser(description='PyTorch training')
parser.add_argument('--apex', action='store_true',
                                       help='User apex for mixed precision training')
parser.add_argument('--apex-opt-level', default='O2', type=str,
                                       help='For apex mixed precision training'
                                                  'O0 for FP32 training, O1 for mixed precison training.')

# 分布式
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='hccl', type=str,
                    help='distributed backend')

## for ascend 910
parser.add_argument('--device_id', default=5, type=int, help='device id')

args = parser.parse_args()
print(args)

# 分布式
rank_size = int(os.environ['RANK_SIZE'])
rank_id = int(os.environ['RANK_ID'])
distributed = rank_size > 1

device = torch.device(f'npu:{args.device_id}')
torch.npu.set_device(device)

if distributed:
    torch.distributed.init_process_group('hccl', rank= rank_id, world_size= rank_size)

# for test
def test(model, test_loader, btrain=False, model_file='model_92.pkl'):
    # Test
    if not btrain:
        model.load_state_dict(torch.load(model_file))
    model.eval()

    correct = 0
    total = 0
    #
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    for images, labels in test_loader:
        #images = Variable(images.cuda())
        images = Variable(images.npu())
        #labels = Variable(labels.cuda())
        labels = Variable(labels.npu())
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()
        #
        c = (predicted == labels.data).squeeze()
        for i in range(20):
            label = labels.data[i]
            class_correct[label] += c[i]
            class_total[label] += 1

    print('Accuracy of the model on the test images: %d %%' % (100 * float(correct) / total))
    print('Accuracy of the model on the test images:', float(correct)/total)
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
    return correct // total


# Image Preprocessing
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop((32, 32), padding=4),   #left, top, right, bottom
    # transforms.Scale(224),
    transforms.ToTensor()
])
test_transform = transforms.Compose([
    transforms.ToTensor()
])
# when image is rgb, totensor do the division 255
# CIFAR-10 Dataset
train_dataset = datasets.CIFAR10(root='./data/',
                               train=True,
                               transform=transform,
                               ##########################
                               download=True)
                               #download=False)
##############################################################

test_dataset = datasets.CIFAR10(root='./data/',
                              train=False,
                              transform=test_transform)

# 分布式 数据切分
if distributed:
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
else:
    train_sampler = None


# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=64, # 64
                                           shuffle=(train_sampler is None), num_workers=8,sampler=train_sampler)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=20,
                                          shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#model = ResidualAttentionModel().cuda()
model = ResidualAttentionModel().npu()
print(model)


if distributed:
    lr = 0.88  
else:
    lr = 0.1  # 0.1
criterion = nn.CrossEntropyLoss()

if args.apex:
    optimizer = apex.optimizers.NpuFusedSGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=0.0001)
    model, optimizer = amp.initialize(model, optimizer, opt_level=args.apex_opt_level, loss_scale=128, combine_grad=True)
else:
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=0.0001)

if distributed:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.device_id],find_unused_parameters=True)

is_train = True
is_pretrain = False
acc_best = 0
#total_epoch = 300
total_epoch = 1
if is_train is True:
    if is_pretrain == True:
        model.load_state_dict((torch.load(model_file)))
    # Training
    for epoch in range(total_epoch):
        if distributed:
            train_sampler.set_epoch(epoch)
        model.train()
        tims = time.time()
        for i, (images, labels) in enumerate(train_loader):
            start_time = time.time()
            #images = Variable(images.cuda())
            images = Variable(images.npu())
            # print(images.data)
            #labels = Variable(labels.cuda())
            labels = Variable(labels.npu())

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if args.apex:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()
            if i < 2:
                print("step_time = {:.4f}".format(time.time() - start_time), flush=True)
            if (i+1) % 100 == 0:
                #print("Epoch [%d/%d], Iter [%d/%d] Loss: %.4f" %(epoch+1, total_epoch, i+1, len(train_loader), loss.data[0]))
                print("Epoch [%d/%d], Iter [%d/%d] Loss: %.4f" %(epoch+1, total_epoch, i+1, len(train_loader), loss.item()))
        print('the epoch takes time:',time.time()-tims)
        print('evaluate test set:')
        acc = test(model, test_loader, btrain=True)
        if acc > acc_best:
            acc_best = acc
            print('current best acc,', acc_best)
            torch.save(model.state_dict(), model_file)
        # Decaying Learning Rate
        if (epoch+1) / float(total_epoch) == 0.3 or (epoch+1) / float(total_epoch) == 0.6 or (epoch+1) / float(total_epoch) == 0.9:
            lr /= 10
            print('reset learning rate to:', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                print(param_group['lr'])
            # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            # optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=0.0001)
    # Save the Model
    torch.save(model.state_dict(), 'last_model_92_sgd.pkl')

else:
    test(model, test_loader, btrain=False)

