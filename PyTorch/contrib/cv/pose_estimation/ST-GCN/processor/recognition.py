#!/usr/bin/env python
# pylint: disable=W0201
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
import sys
import argparse
import yaml
import numpy as np
import time
# torch
import torch
import torch.nn as nn
import torch.optim as optim

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class
from apex import amp, optimizers
try:
    from torch_npu.utils.profiler import Profile
except ImportError:
    print("Profile not in torch_npu.utils.profiler now... Auto Profile disabled.", flush=True)
    class Profile:
        def __init__(self, *args, **kwargs):
            pass
        def start(self):
            pass
        def end(self):
            pass

from .processor import Processor


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', start_count_index=5):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.start_count_index = start_count_index

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if self.count == 0:
            self.N = n

        self.val = val
        self.count += n
        if self.count > (self.start_count_index * self.N):
            self.sum += val * n
            self.avg = self.sum / \
                (self.count - self.start_count_index * self.N)

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class REC_Processor(Processor):
    """
        Processor for Skeleton-based Action Recgnition
    """

    def load_model(self):
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))
        self.model.apply(weights_init)
        self.loss = nn.CrossEntropyLoss()

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

    def adjust_lr(self):
        if self.arg.optimizer == 'SGD' and self.arg.step:
            if self.arg.warm_up:
                if self.meta_info['epoch'] < self.arg.warm_up_epochs:
                    lr = 0.2 + \
                        self.meta_info['epoch'] * \
                        (self.arg.base_lr - 0.2)/(self.arg.warm_up_epochs+1)

                else:
                    lr = self.arg.base_lr * (
                        0.1**np.sum(self.meta_info['epoch'] >= np.array(self.arg.step)))
            else:
                lr = self.arg.base_lr * (
                    0.1**np.sum(self.meta_info['epoch'] >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr

    def show_topk(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        if self.rank == 0 or self.world_size == 1:
            self.io.print_log('\tTop{}: {:.2f}%'.format(k, 100 * accuracy))
            if accuracy > self.best_acc and k == 1 and self.arg.phase == 'train':
                filename = 'best_model_{}p.pt'.format(self.world_size)
                self.io.save_model(self.model, filename)
                self.io.print_log(
                    '\tBest Top1: {:.2f}%,saved as best_model.pt'.format(100 * accuracy))
                self.best_acc = accuracy
                self.io.print_log(
                    "Temp Best Acc is : {:.2f}%".format(100*self.best_acc))

    def train(self):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []
        batch_time = AverageMeter('Time', ':6.3f')
        end = time.time()
        profiler = Profile(start_step=int(os.getenv("PROFILE_START_STEP", 10)),
                           profile_type=os.getenv("PROFILE_TYPE"))
        for data, label in loader:
            if self.meta_info['iter'] > self.arg.steps_per_epoch:
                continue
            if self.arg.profiling != 'NONE' and self.meta_info['iter'] >=self.arg.stop_step:
                import sys
                sys.exit()

            # get data
            data = data.float().to(self.dev)
            data.requires_grad = True
            if "gpu" in self.arg.use_gpu_npu:
                label = label.long().to(self.dev)
            else:
                label = label.int().to(self.dev)

            if self.meta_info['iter'] <= self.arg.stop_step and self.meta_info['iter'] >= self.arg.start_step \
                    and self.arg.profiling == 'CANN':
                with torch.npu.profile(profiler_result_path="./CANN_prof"):
                    # forward
                    start_time = time.time()
                    output = self.model(data)
                    loss = self.loss(output, label)
                    # backward
                    self.optimizer.zero_grad()
                    if self.amp:
                        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                    # loss.backward()
                    self.optimizer.step()
                    # statistics
                    self.iter_info['loss'] = loss.data.item()
                    self.iter_info['lr'] = '{:.6f}'.format(self.lr)
                    loss_value.append(self.iter_info['loss'])
                    train_time = time.time() - start_time
                    self.iter_info['s/step'] = train_time
                    self.show_iter_info()
                    self.meta_info['iter'] += 1
                    batch_time.update(time.time() - end)
                    end = time.time()

            elif self.meta_info['iter'] <= self.arg.stop_step and self.meta_info['iter'] >= self.arg.start_step \
                    and self.arg.profiling == 'GE':
                with torch.npu.profile(profiler_result_path="./GE_prof"):
                    # forward
                    start_time = time.time()
                    output = self.model(data)
                    loss = self.loss(output, label)
                    # backward
                    self.optimizer.zero_grad()
                    if self.amp:
                        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                    # loss.backward()
                    self.optimizer.step()
                    # statistics
                    self.iter_info['loss'] = loss.data.item()
                    self.iter_info['lr'] = '{:.6f}'.format(self.lr)
                    loss_value.append(self.iter_info['loss'])
                    train_time = time.time() - start_time
                    self.iter_info['s/step'] = train_time
                    self.show_iter_info()
                    self.meta_info['iter'] += 1
                    batch_time.update(time.time() - end)
                    end = time.time()

            else:
                profiler.start()
                # forward
                start_time = time.time()
                output = self.model(data)
                loss = self.loss(output, label)
                # backward
                self.optimizer.zero_grad()
                if self.amp:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                # loss.backward()
                self.optimizer.step()
                # statistics
                self.iter_info['loss'] = loss.data.item()
                self.iter_info['lr'] = '{:.6f}'.format(self.lr)
                loss_value.append(self.iter_info['loss'])
                train_time = time.time() - start_time
                self.iter_info['s/step'] = train_time
                self.show_iter_info()
                self.meta_info['iter'] += 1
                batch_time.update(time.time() - end)
                end = time.time()
                profiler.end()

        self.epoch_info['mean_loss'] = np.mean(loss_value)
        self.show_epoch_info()
        self.io.print_timer()
        if batch_time.avg > 0 and (self.rank == 0 or self.world_size == 1):
            self.io.print_log(
                "[gpu num:{} ],* FPS@all {:.3f}, TIME@all {:.3f}".format(
                    self.world_size,
                    self.world_size * self.arg.batch_size / batch_time.avg,
                    batch_time.avg))

    def test(self, evaluation=True):

        self.model.eval()
        loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []

        for data, label in loader:

            # get data
            data = data.float().to(self.dev)
            if "gpu" in self.arg.use_gpu_npu:
                label = label.long().to(self.dev)
            else:
                label = label.int().to(self.dev)

            # inference
            with torch.no_grad():
                output = self.model(data)
            result_frag.append(output.data.cpu().numpy())

            # get loss
            if evaluation:
                loss = self.loss(output, label)
                loss_value.append(loss.item())
                label_frag.append(label.data.cpu().numpy())

        self.result = np.concatenate(result_frag)
        if evaluation:
            self.label = np.concatenate(label_frag)
            self.epoch_info['mean_loss'] = np.mean(loss_value)
            self.show_epoch_info()

            # show top-k accuracy
            for k in self.arg.show_topk:
                self.show_topk(k)

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        # evaluation
        parser.add_argument('--show_topk', type=int,
                            default=[1, 5], nargs='+', help='which Top K accuracy will be shown')
        # optim
        parser.add_argument('--base_lr', type=float,
                            default=0.01, help='initial learning rate')
        parser.add_argument('--warm_up', type=str2bool,
                            default=False, help='warmup or not')
        parser.add_argument('--warm_up_epochs', type=int,
                            default=5, help='warmup epochs')
        parser.add_argument('--step', type=int, default=[], nargs='+',
                            help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD',
                            help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool,
                            default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float,
                            default=0.0001, help='weight decay for optimizer')
        # endregion yapf: enable

        return parser
