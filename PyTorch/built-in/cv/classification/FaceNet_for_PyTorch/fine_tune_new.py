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

import torch
import sys
import os
import random
import argparse
import apex
import numpy as np
import torch.backends.cudnn as cudnn
import torchvision.models as models
from models.inception_resnet_v1 import InceptionResnetV1
from models.mtcnn import MTCNN, fixed_image_standardization
from models.utils import training
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
# from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from apex import amp

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def parse_opts():
    parser = argparse.ArgumentParser(description='facenet')
    parser.add_argument('--seed', type=int, default=123456, help='random seed')
    parser.add_argument('--amp_cfg', action='store_true',
                        help='If true, use apex.')
    parser.add_argument('--opt_level', default='O0', type=str,
                        help='set opt level.')
    parser.add_argument('-a', '--arch', metavar='ARCH',
                        default='inceptionresnetv1', choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet18)')
    parser.add_argument('--fine_tuning', action='store_true',
                        help='use fine-tuning model')
    parser.add_argument('--loss_scale_value', default=1024, type=int,
                        help='set loss scale value.')
    parser.add_argument('--device_list', default='0,1,2,3,4,5,6,7', type=str,
                        help='device id list')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='set batch_size')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--epochs', default=20, type=int, help='set epochs')
    parser.add_argument('--epochs_per_save', default=1, type=int,
                        help='save per epoch')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--workers', default=0, type=int, help='set workers')
    parser.add_argument('--data_dir', default="", type=str,
                        help='set data_dir')
    args = parser.parse_args()
    return args


def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    args = parse_opts()
    seed_everything(args.seed)
    device_id = int(args.device_list.split(",")[0])
    device = 'npu:{}'.format(device_id)
    print('Running on device: {}'.format(device))
    torch.npu.set_device(device)
    dataset = datasets.ImageFolder(args.data_dir, transform=None)

    if args.fine_tuning:
        print(
            "=> transfer-learning mode + fine-tuning"
            "(train only the last FC layer)")
        if args.arch == 'inceptionresnetv1':
            print("=> Fine_tune on this casia-webface dataset")
            resnet = InceptionResnetV1(
                classify=True,
                pretrained='casia-webface',
                num_classes=len(dataset.class_to_idx)).to(device)
        else:
            print("Error:Fine-tuning is not supported on this architecture")
            exit(-1)
    else:
        print("=> using pre-trained model '{}'".format(args.arch))
        resnet = InceptionResnetV1(
            classify=True,
            pretrained='vggface2',
            num_classes=len(dataset.class_to_idx)).to(device)

    optimizer = apex.optimizers.NpuFusedAdam(resnet.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, [5, 10])
    if args.amp_cfg:
        if args.resume and args.opt_level == 'O2':
            args.opt_level = 'O1'
        resnet, optimizer = amp.initialize(resnet, optimizer,
                                           opt_level=args.opt_level,
                                           loss_scale=args.loss_scale_value, combine_grad=True)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=device)
            args.start_epoch = checkpoint['epoch']
            resnet.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            if args.amp_cfg:
                amp.load_state_dict(checkpoint['amp'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    trans = transforms.Compose([
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization
    ])
    dataset = datasets.ImageFolder(args.data_dir, transform=trans)
    img_inds = np.arange(len(dataset))

    np.random.shuffle(img_inds)
    train_inds = img_inds[:int(0.8 * len(img_inds))]
    val_inds = img_inds[int(0.8 * len(img_inds)):]

    train_loader = DataLoader(
        dataset,
        num_workers=args.workers,
        batch_size=args.batch_size,
        sampler=SubsetRandomSampler(train_inds)
    )
    val_loader = DataLoader(
        dataset,
        num_workers=args.workers,
        batch_size=args.batch_size,
        sampler=SubsetRandomSampler(val_inds)
    )

    loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn.to(device)

    metrics = {
        'fps': training.BatchTimer(),
        'acc': training.accuracy
    }

    # writer = SummaryWriter()
    # writer.iteration, writer.interval = 0, 10

    print('\n\nInitial')
    print('-' * 10)

    resnet.eval()
    training.pass_epoch(
        args.amp_cfg, resnet, loss_fn, val_loader,
        batch_metrics=metrics, show_running=True, device=device,
        # writer=writer
    )

    for epoch in range(args.start_epoch, args.epochs):
        print('\nEpoch {}/{}'.format(epoch + 1, args.epochs))
        print('-' * 10)

        resnet.train()
        training.pass_epoch(
            args.amp_cfg, resnet, loss_fn, train_loader, optimizer, scheduler,
            batch_metrics=metrics, show_running=True, device=device,
            # writer=writer
        )
        if (epoch + 1) % args.epochs_per_save == 0 or epoch + 1 == args.epochs:
            if not os.path.isdir("./model_param"):
                os.mkdir("./model_param")

            if args.amp_cfg:
                torch.save({'epoch': epoch + 1,
                            'net': resnet.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler':scheduler.state_dict(),
                            'amp': amp.state_dict()},
                            './model_param/checkpoint_epoch%d.pth'
                           % (epoch + 1))
            else:
                torch.save({'epoch': epoch + 1,
                            'net': resnet.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler':scheduler.state_dict()},
                            './model_param/checkpoint_epoch%d.pth'
                           % (epoch + 1))

        resnet.eval()
        training.pass_epoch(
            args.amp_cfg, resnet, loss_fn, val_loader,
            batch_metrics=metrics, show_running=True, device=device,
            # writer=writer
        )

    # writer.close()


if __name__ == "__main__":
    main()












