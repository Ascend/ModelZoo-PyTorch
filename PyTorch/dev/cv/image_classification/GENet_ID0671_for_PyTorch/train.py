#BSD 3-Clause License
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
#Copyright (c) 2017, 
#All rights reserved.

#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:

#* Redistributions of source code must retain the above copyright notice, this
#  list of conditions and the following disclaimer.

#* Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.

#* Neither the name of the copyright holder nor the names of its
#  contributors may be used to endorse or promote products derived from
#  this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#================================================================================

import argparse
import json
import os
import time
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.model_zoo as model_zoo
from tensorboardX import SummaryWriter
import torchvision
import torchvision.transforms as transforms
from models import *
from utils import *
import torch
from apex import amp
import apex

writer = SummaryWriter()

parser = argparse.ArgumentParser(description='Pruning')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--GPU', default='3', type=str, help='GPU to use')
parser.add_argument('--save_file', default='saveto', type=str, help='save file for checkpoints')
parser.add_argument('--print_freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')

# Learning specific arguments
parser.add_argument('-b', '--batch_size', default=128, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-bt', '--test_batch_size', default=128, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--device_id', default=5, type=int, help='device id')
parser.add_argument('-lr', '--learning_rate', default=.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--lr_decay_ratio', default=0.2, type=float, help='learning rate decay factor')
parser.add_argument('-epochs', '--no_epochs', default=1, type=int, metavar='epochs', help='no. epochs')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--nesterov', default=False, type=bool, help='yesterov?')
parser.add_argument('--weight_decay', '--wd', default=0.0005, type=float, metavar='W', help='weight decay')
parser.add_argument('--eval', '-e', action='store_true', help='resume from checkpoint')
parser.add_argument('--dataset', choices=['cifar10','cifar100'], default = 'cifar10')
parser.add_argument('--epoch_step', default='[60,120,160]', type=str, help='json list with epochs to drop lr on')

# Mixed precision training parameters
parser.add_argument('--apex', action='store_true',
                    help='Use apex for mixed precision training')
parser.add_argument('--apex-opt-level', default='O1', type=str,
                    help='For apex mixed precision training'
                         'O0 for FP32 training, O1 for mixed precision training.'
                         'For further detail, see https://github.com/NVIDIA/apex/tree/master/examples/imagenet')
parser.add_argument('--loss-scale-value', default=1024., type=float,
                    help='loss scale using in amp, default -1 means dynamic')


#Net specific
parser.add_argument('--depth', '-d', default=16, type=int, metavar='D', help='wrn depth')
parser.add_argument('--width', '-w', default=8, type=int, metavar='W', help='wrn width)')
parser.add_argument('--mlp', default=False, type=bool, help='mlp?')
parser.add_argument('--extra_params', default=False, type=bool, help='extraparams?')
parser.add_argument('--extent', default=0, type=int, help='Extent for pooling')


args = parser.parse_args()
print(args)
os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device(f'npu:{args.device_id}' if torch.npu.is_available() else "cpu")
torch.npu.set_device(device)
print("***Device ID:", args.device_id)

if not os.path.exists('checkpoints/'):
    os.makedirs('checkpoints/')

# GET NORMS FOR DATASET

if args.dataset == 'cifar10':
    MEAN = (0.4914, 0.4822, 0.4465)
    STD  = (0.2023, 0.1994, 0.2010)
    NO_CLASSES = 10

elif args.dataset == 'cifar100':
    MEAN =  (0.5071, 0.4867, 0.4408)
    STD =    (0.2675, 0.2565, 0.2761)
    NO_CLASSES = 100
else:
    raise ValueError('pick a dataset')


model = WideResNet(args.depth, args.width, num_classes=NO_CLASSES, mlp=args.mlp, extra_params=args.extra_params)

get_no_params(model)
model.to(device)


print('Standard Aug')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN,STD),
])


transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])




if args.dataset == 'cifar10':
    trainset = torchvision.datasets.CIFAR10(root='data/cifar10',
                                            train=True, download=False, transform=transform_train)



    valset = torchvision.datasets.CIFAR10(root='data/cifar10',
                                          train=False, download=False, transform=transform_val)

elif args.dataset == 'cifar100':
    trainset = torchvision.datasets.CIFAR100(root='/disk/scratch/datasets/cifar',
                                            train=True, download=False, transform=transform_train)
    valset = torchvision.datasets.CIFAR100(root='/disk/scratch/datasets/cifar',
                                          train=False, download=False, transform=transform_val)

else:
    raise ValueError('Pick a dataset (ii)')


trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                          num_workers=args.workers,
                                          pin_memory=False,
                                          shuffle=True)

valloader = torch.utils.data.DataLoader(valset, batch_size=args.test_batch_size, shuffle=False,
                                        num_workers=args.workers,
                                        pin_memory=False)

error_history = []
epoch_step = json.loads(args.epoch_step)

def train():
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for i, (input, target) in enumerate(trainloader):

        # measure data loading time
        data_time.update(time.time() - end)

        #input, target = input.to(device), target.to(device)
        input, target = input.to(device), target.to(torch.int).to(device)

        output = model(input)

        loss = criterion(output, target)

        # measure accuracy and record loss
        err1, err5 = get_error(output.detach(), target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if args.apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
		##        
        batch_sise = target.size(0)
        fps = (batch_sise / batch_time.val)

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'FPS {fps:.3f}\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Error@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(trainloader), batch_time=batch_time,fps=fps,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

    writer.add_scalar('train_loss', losses.avg, epoch)
    writer.add_scalar('train_top1', top1.avg, epoch)
    writer.add_scalar('train_top5', top5.avg, epoch)


def validate():
    global error_history

    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    for i, (input, target) in enumerate(valloader):

        # measure data loading time
        data_time.update(time.time() - end)

        #input, target = input.to(device), target.to(device)
        input, target = input.to(device), target.to(torch.int).to(device)

        # compute output
        output = model(input)

        loss = criterion(output, target)

        # measure accuracy and record loss
        err1, err5 = get_error(output.detach(), target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Error@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(valloader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

    print(' * Error@1 {top1.avg:.3f} Error@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    writer.add_scalar('val_loss', losses.avg, epoch)
    writer.add_scalar('val_top1', top1.avg, epoch)
    writer.add_scalar('val_top5', top5.avg, epoch)

    # Record Top 1 for CIFAR
    error_history.append(top1.avg)


if __name__ == '__main__':

    filename = 'checkpoints/%s.t7' % args.save_file
    criterion = nn.CrossEntropyLoss()
    optimizer = apex.optimizers.NpuFusedSGD(model.parameters(),
                                lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay,
                                nesterov=args.nesterov)

    if args.apex:
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=args.apex_opt_level,
                                          loss_scale=args.loss_scale_value,
                                          combine_grad=True)

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=epoch_step, gamma=args.lr_decay_ratio)



    if not args.eval:

        for epoch in range(args.no_epochs):
            scheduler.step()

            print('Epoch %d:' % epoch)
            print('Learning rate is %s' % [v['lr'] for v in optimizer.param_groups][0])
            # train for one epoch
            train()
            # # # evaluate on validation set
            if 1:
                validate()
                #
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'error_history': error_history,
                }, filename=filename)

    else:
        if not args.deploy:
            model.load_state_dict(torch.load(filename)['state_dict'])
        epoch = 0
        validate()
