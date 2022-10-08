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
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
if torch.__version__ >= "1.8":
    import torch_npu
import torch.backends.cudnn as cudnn
import torchvision
import apex
from apex import amp
from model import Net
import torch.npu
import os

NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')

parser = argparse.ArgumentParser(description="Train on market1501")
parser.add_argument("--data-dir", default='data', type=str)
# parser.add_argument("--no-cuda",action="store_true")
parser.add_argument("--no-npu", action="store_true")
parser.add_argument("--gpu-id", default=0, type=int)
parser.add_argument("--lr", default=0.1, type=float)
parser.add_argument("--interval", '-i', default=20, type=int)
parser.add_argument('--resume', '-r', action='store_true')
parser.add_argument('--ddp', action='store_true', help='default 1p')
# parser.add_argument('--device_id', default=5, type=int, help='device_id')
parser.add_argument('--apex', action='store_true',
                    help='User apex for mixed precision training')
parser.add_argument('--apex-opt-level', default='O1', type=str,
                    help='For apex mixed precision training'
                         'O0 for FP32 training, O1 for mixed precison training.')
parser.add_argument('--loss-scale-value', default=1024., type=float,
                    help='loss scale using in amp, default -1 means dynamic')
parser.add_argument('--max_steps', default=None, type=int, metavar='N', help='number of total steps to run')
args = parser.parse_args()

if args.ddp:
    NPU_WORLD_SIZE = int(os.getenv('NPU_WORLD_SIZE'))
    RANK = int(os.getenv('RANK'))
    torch.distributed.init_process_group('hccl', rank=RANK, world_size=NPU_WORLD_SIZE)

# device
# device = "cuda:{}".format(args.gpu_id) if torch.cuda.is_available() and not args.no_cuda else "cpu"
device = torch.device(f'npu:{NPU_CALCULATE_DEVICE}') if torch.npu.is_available() and not args.no_npu else "cpu"
torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')
# if torch.cuda.is_available() and not args.no_cuda:
if torch.npu.is_available() and not args.no_npu:
    # cudnn.benchmark = True
    cudnn.benchmark = False

# data loading
root = args.data_dir
train_dir = os.path.join(root, "train")
test_dir = os.path.join(root, "test")
transform_train = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop((128, 64), padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize((128, 64)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
if args.ddp:
    trainloader_sampler = torch.utils.data.distributed.DistributedSampler(
        torchvision.datasets.ImageFolder(train_dir, transform=transform_train))
    trainloader_batch_size = 64
    trainloader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(train_dir, transform=transform_train),
        batch_size=trainloader_batch_size, shuffle=False,
        pin_memory=True, drop_last=True, sampler=trainloader_sampler)
#    testloader_sampler = torch.utils.data.distributed.DistributedSampler(
#        torchvision.datasets.ImageFolder(test_dir, transform=transform_test))
#    testloader_batch_size = 64
#    testloader = torch.utils.data.DataLoader(
#        torchvision.datasets.ImageFolder(test_dir, transform=transform_test),
#        batch_size=testloader_batch_size, shuffle=False,
#        pin_memory=True, drop_last=True, sampler=testloader_sampler)
else:
    trainloader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(train_dir, transform=transform_train),
        batch_size=64, shuffle=True, num_workers=96
    )
testloader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(test_dir, transform=transform_test),
        batch_size=64, shuffle=True, num_workers=96
    )
num_classes = max(len(trainloader.dataset.classes), len(testloader.dataset.classes))

# net definition
start_epoch = 0
net = Net(num_classes=num_classes)
if args.resume:
    assert os.path.isfile("./checkpoint/ckpt.t7"), "Error: no checkpoint file found!"
    print('Loading from checkpoint/ckpt.t7')
    checkpoint = torch.load("./checkpoint/ckpt.t7")
    # import ipdb; ipdb.set_trace()
    net_dict = checkpoint['net_dict']
    net.load_state_dict(net_dict)
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
net.to(f'npu:{NPU_CALCULATE_DEVICE}')

# loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
if args.apex:
    optimizer = apex.optimizers.NpuFusedSGD(net.parameters(), args.lr, momentum=0.9, weight_decay=5e-4)
else:
    optimizer = torch.optim.SGD(net.parameters(), args.lr, momentum=0.9, weight_decay=5e-4)

if args.apex:
    net, optimizer = amp.initialize(net, optimizer, opt_level=args.apex_opt_level,
                                    loss_scale=args.loss_scale_value,
                                    combine_grad=True)
if args.ddp:
    net = net.to(f'npu:{NPU_CALCULATE_DEVICE}')
    if not isinstance(net, torch.nn.parallel.DistributedDataParallel):
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[NPU_CALCULATE_DEVICE], broadcast_buffers=False)
best_acc = 0.


# train function for each epoch
def train(epoch):
    print("\nEpoch : %d" % (epoch + 1))
    net.train()
    training_loss = 0.
    train_loss = 0.
    correct = 0
    total = 0
    interval = args.interval
    start = time.time()
    for idx, (inputs, labels) in enumerate(trainloader):
        if args.max_steps and idx >= args.max_steps:
            break
        # forward
        #with torch.autograd.profiler.profile(use_npu=True) as prof:
        inputs, labels = inputs.to(f'npu:{NPU_CALCULATE_DEVICE}'), labels.to(f'npu:{NPU_CALCULATE_DEVICE}')
        outputs = net(inputs)
        loss = criterion(outputs, labels)

            # backward
        optimizer.zero_grad()
        if args.apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        #prof.export_chrome_trace(f'./resnet_{NPU_CALCULATE_DEVICE}.json')
        # accumurating
        training_loss += loss.item()
        train_loss += loss.item()
        correct += outputs.max(dim=1)[1].eq(labels).sum().item()
        total += labels.size(0)

        # print 
        if (idx + 1) % interval == 0:
            end = time.time()
            print("[progress:{:.1f}%] time:{:.2f}s Loss:{:.5f} Correct:{}/{} Acc:{:.3f}%".format(
                100. * (idx + 1) / len(trainloader), end - start, training_loss / interval, correct, total,
                100. * correct / total
            ))
            training_loss = 0.
            start = time.time()
    return train_loss / len(trainloader), 1. - correct / total
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0.
    correct = 0
    total = 0
    start = time.time()
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(testloader):
            if args.max_steps and idx >= args.max_steps:
                break
            inputs, labels = inputs.to(f'npu:{NPU_CALCULATE_DEVICE}'), labels.to(f'npu:{NPU_CALCULATE_DEVICE}')
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            correct += outputs.max(dim=1)[1].eq(labels).sum().item()
            total += labels.size(0)

        print("Testing ...")
        end = time.time()
        print("[progress:{:.1f}%] time_test:{:.2f}s Loss:{:.5f} Correct:{}/{} Acc:{:.3f}%".format(
            100. * (idx + 1) / len(testloader), end - start, test_loss / len(testloader), correct, total,
            100. * correct / total
        ))

    # saving checkpoint
    acc = 100. * correct / total
    if acc > best_acc:
        best_acc = acc
        print("Saving parameters to checkpoint/ckpt.t7")
        checkpoint = {
            'net_dict': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(checkpoint, './checkpoint/ckpt.t7')

    return test_loss / len(testloader), 1. - correct / total


# plot figure
x_epoch = []
record = {'train_loss': [], 'train_err': [], 'test_loss': [], 'test_err': []}
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")


def draw_curve(epoch, train_loss, train_err, test_loss, test_err):
    global record
    record['train_loss'].append(train_loss)
    record['train_err'].append(train_err)
    record['test_loss'].append(test_loss)
    record['test_err'].append(test_err)

    x_epoch.append(epoch)
    ax0.plot(x_epoch, record['train_loss'], 'bo-', label='train')
    ax0.plot(x_epoch, record['test_loss'], 'ro-', label='val')
    ax1.plot(x_epoch, record['train_err'], 'bo-', label='train')
    ax1.plot(x_epoch, record['test_err'], 'ro-', label='val')
    if epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig("train.jpg")


# lr decay
def lr_decay():
    global optimizer
    for params in optimizer.param_groups:
        params['lr'] *= 0.1
        lr = params['lr']
        print("Learning rate adjusted to {}".format(lr))


def main():
    for epoch in range(start_epoch, start_epoch + 40):
        # for epoch in range(start_epoch, start_epoch+40):
        if args.ddp:
            trainloader.sampler.set_epoch(epoch)
            #testloader.sampler.set_epoch(epoch)
        train_loss, train_err = train(epoch)
        test_loss, test_err = test(epoch)
        draw_curve(epoch, train_loss, train_err, test_loss, test_err)
        if (epoch + 1) % 20 == 0:
            lr_decay()


if __name__ == '__main__':
    main()
