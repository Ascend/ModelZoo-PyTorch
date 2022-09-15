# -*- coding: utf-8 -*-
# BSD 3-Clause License
#
# Copyright (c) 2017, 
# All rights reserved.
#
# Copyright 2021 Huawei Technologies Co., Ltd
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
# limitations under the License


import os
import re
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from multi_epochs_dataloader import MultiEpochsDataLoader

class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res


def save_checkpoint(state, iters, tag=''):
    if not os.path.exists("./models"):
        os.makedirs("./models")
    filename = os.path.join("./models/{}checkpoint-{:06}.pth.tar".format(tag, iters))
    torch.save(state, filename)

def get_lastest_model():
    if not os.path.exists('./models'):
        os.mkdir('./models')
    model_list = os.listdir('./models/')
    if model_list == []:
        return None, 0
    model_list.sort()
    lastest_model = model_list[-1]
    iters = re.findall(r'\d+', lastest_model)
    return './models/' + lastest_model, int(iters[0])


def get_parameters(model):
    group_no_weight_decay = []
    group_weight_decay = []
    for pname, p in model.named_parameters():
        if pname.find('weight') >= 0 and len(p.size()) > 1:
            # print('include ', pname, p.size())
            group_weight_decay.append(p)
        else:
            # print('not include ', pname, p.size())
            group_no_weight_decay.append(p)
    assert len(list(model.parameters())) == len(group_weight_decay) + len(group_no_weight_decay)
    groups = [dict(params=group_weight_decay), dict(params=group_no_weight_decay, weight_decay=0.)]
    return groups


def get_pytorch_train_loader(data_path, batch_size, workers=5, _worker_init_fn=None, distributed=False):
    traindir = os.path.join(data_path, 'train')
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip()
        ]))

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    dataloader_fn = MultiEpochsDataLoader
    train_loader = dataloader_fn(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=workers, worker_init_fn=_worker_init_fn, pin_memory=False, sampler=train_sampler,
        collate_fn=fast_collate, drop_last=True)

    return train_loader, len(train_loader), train_sampler


def get_pytorch_val_loader(data_path, batch_size, workers=5, _worker_init_fn=None, distributed=False):
    valdir = os.path.join(data_path, 'val')
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]))

    if distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        val_sampler = None

    dataloader_fn = MultiEpochsDataLoader
    val_loader = dataloader_fn(
        val_dataset,
        sampler=val_sampler,
        batch_size=batch_size,
        shuffle=(val_sampler is None),
        num_workers=workers, worker_init_fn=_worker_init_fn,
        pin_memory=False, collate_fn=fast_collate, drop_last=True)

    return val_loader


def fast_collate(batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if nump_array.ndim < 3:
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)
        tensor[i] += torch.from_numpy(nump_array)

    return tensor, targets


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = args.lr * (0.1 ** (epoch // (args.epochs//3 - 3)))

    if args.warm_up_epochs > 0 and epoch < args.warm_up_epochs:
        lr = args.learning_rate * ((epoch + 1) / (args.warm_up_epochs + 1))
    else:
        alpha = 0
        cosine_decay = 0.5 * (
                1 + np.cos(np.pi * (epoch - args.warm_up_epochs) / (args.epochs - args.warm_up_epochs)))
        decayed = (1 - alpha) * cosine_decay + alpha
        lr = args.learning_rate * decayed

    print("=> Epoch[%d] Setting lr: %.4f" % (epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr