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

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import time
import torch
import argparse
import torch.optim as optim
import torch.utils.data as data
import numpy as np
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from apex import amp

from data.config import cfg
from s3fd import build_s3fd
from layers.modules import MultiBoxLoss
from data.factory import dataset_factory, detection_collate
# Apex imports
try:
    import apex_C
    import apex
    from apex.parallel.LARC import LARC
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex.multi_tensor_apply import multi_tensor_applier
    #import amp_C
except ImportError:
    raise ImportError("Please install APEX from https://github.com/nvidia/apex")

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='S3FD face Detector Training With Pytorch')
parser.add_argument('--dataset',
                    default='face',
                    choices=['hand', 'face', 'head'],
                    help='Train target')
parser.add_argument('--basenet',
                    default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size',
                    default=16, type=int,
                    help='Batch size for training')
parser.add_argument('--resume',
                    default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--num_workers',
                    default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate',
                    default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum',
                    default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay',
                    default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma',
                    default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--multigpu',
                    default=False, type=str2bool,
                    help='Use mutil Gpu training')
parser.add_argument('--save_folder',
                    default='weights/',
                    help='Directory for saving checkpoint models')
# for ascend 910
parser.add_argument('--device',
                    default='npu', type=str,
                    help='Use npu or gpu or cpu to train model')
parser.add_argument('--device_id', default=0, type=int, help='device id')
# parser.add_argument('--addr', default='10.136.181.115',
#                     type=str, help='master addr')

# for modelArts
# parser.add_argument('--device_list', default='0,1,2,3,4,5,6,7',
#                     type=str, help='device id list')
# parser.add_argument('--warm_up_epochs', default=0, type=int,
#                     help='warm up')
parser.add_argument('--amp', default=False, action='store_true',
                    help='use amp to train the model')
parser.add_argument('--loss-scale', default=-1, type=float,
                    help='loss scale using in amp, default -1 means dynamic')

args = parser.parse_args()

device = ''
if args.device == 'gpu':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
elif args.device == 'npu':
    device = 'npu:{}'.format(args.device_id)
    torch.npu.set_device(device)
    print("Use NPU: {} for training".format(args.device_id))

if args.device == 'gpu' and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder)

train_dataset, val_dataset = dataset_factory(args.dataset)

train_loader = data.DataLoader(train_dataset, args.batch_size,
                               num_workers=args.num_workers,
                               shuffle=True,
                               collate_fn=detection_collate,
                               pin_memory=True)

val_batchsize = args.batch_size // 2
val_loader = data.DataLoader(val_dataset, val_batchsize,
                             num_workers=args.num_workers,
                             shuffle=False,
                             collate_fn=detection_collate,
                             pin_memory=True)

min_loss = np.inf
start_epoch = 0
s3fd_net = build_s3fd('train', cfg.NUM_CLASSES, args.device)
net = s3fd_net

if args.resume:
    print('Resuming training, loading {}...'.format(args.resume))
    start_epoch = net.load_weights(args.resume)

else:
    vgg_weights = torch.load(args.save_folder + args.basenet)
    print('Load base network....')
    net.vgg.load_state_dict(vgg_weights)

if args.device != 'cpu':
    if args.multigpu:
        net = torch.nn.DataParallel(s3fd_net)
    net = net.to(device)
    cudnn.benckmark = True

if not args.resume:
    print('Initializing weights...')
    s3fd_net.extras.apply(s3fd_net.weights_init)
    s3fd_net.loc.apply(s3fd_net.weights_init)
    s3fd_net.conf.apply(s3fd_net.weights_init)

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)

if args.amp:
    net, optimizer = amp.initialize(
        net, optimizer, opt_level='O1', combine_grad=True)

criterion = MultiBoxLoss(cfg, args.dataset, args.device)
print('Loading wider dataset...')
print('Using the specified args:')
print(args)


def train():
    """
    train
    """
    step_index = 0
    iteration = 0
    net.train()
    for epoch in range(start_epoch, cfg.EPOCHES):
        losses = 0
        for batch_idx, (images, targets) in enumerate(train_loader):
            if args.device != 'cpu':
                images = Variable(images.to(device))
                targets = [Variable(ann.to(device), volatile=True)
                           for ann in targets]
            else:
                images = Variable(images)
                targets = [Variable(ann, volatile=True) for ann in targets]

            if iteration in cfg.LR_STEPS:
                step_index += 1
                adjust_learning_rate(optimizer, args.gamma, step_index)

            t0 = time.time()
            out = net(images)

            # backprop
            optimizer.zero_grad()

            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c

            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            optimizer.step()
            t1 = time.time()

            if args.device == 'npu':
                losses += loss.cpu().item()
            else:
                losses += loss.item()

            if iteration % 10 == 0:
                tloss = losses / (batch_idx + 1)
                print('Timer: %.4f' % (t1 - t0))
                print('epoch:' + repr(epoch) + ' || iter:' +
                      repr(iteration) + ' || Loss:%.4f' % (tloss))
                print('->> conf loss:{:.4f} || loc loss:{:.4f}'.format(
                    loss_c.item(), loss_l.item()))
                print('->>lr:{:.6f}'.format(optimizer.param_groups[0]['lr']))

            if iteration != 0 and iteration % 5000 == 0:
                print('Saving state, iter:', iteration)
                file = 'sfd_' + args.dataset + '_' + repr(iteration) + '.pth'
                torch.save(s3fd_net.state_dict(),
                           os.path.join(args.save_folder, file))
            iteration += 1

        val(epoch)
        if iteration == cfg.MAX_STEPS:
            break


def val(epoch):
    """
    val
    """
    net.eval()
    loc_loss = 0
    conf_loss = 0
    step = 0
    t1 = time.time()
    for batch_idx, (images, targets) in enumerate(val_loader):
        if args.device != 'cpu':
            images = Variable(images.to(device))
            targets = [Variable(ann.to(device), volatile=True)
                       for ann in targets]
        else:
            images = Variable(images)
            targets = [Variable(ann, volatile=True) for ann in targets]

        out = net(images)
        loss_l, loss_c = criterion(out, targets)

        loc_loss += loss_l.item()
        conf_loss += loss_c.item()
        step += 1

    tloss = (loc_loss + conf_loss) / step
    t2 = time.time()
    print('Timer: %.4f' % (t2 - t1))
    print('test epoch:' + repr(epoch) + ' || Loss:%.4f' % (tloss))

    global min_loss
    if tloss < min_loss:
        print('Saving best state,epoch', epoch)
        file = 'sfd_{}.pth'.format(args.dataset)
        torch.save(s3fd_net.state_dict(), os.path.join(
            args.save_folder, file))
        min_loss = tloss

    states = {
        'epoch': epoch,
        'weight': s3fd_net.state_dict(),
    }
    file = 'sfd_{}_checkpoint.pth'.format(args.dataset)
    torch.save(states, os.path.join(
        args.save_folder, file))


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    train()
