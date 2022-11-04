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
import sys
import argparse
import yaml
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import os
# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .io import IO
import torch.distributed as dist
from apex import amp, optimizers


class Processor(IO):
    """
        Base Processor
    """

    def __init__(self, argv=None):

        self.load_arg(argv)
        self.world_size = len([self.arg.device] if isinstance(
            self.arg.device, int) else list(self.arg.device))
        os.environ['world_size'] = str(self.world_size)
        self.amp = self.arg.amp

    def parallel_train(self, rank):
        torch.manual_seed(1)
        self.rank = rank
        self.init_environment()
        if self.world_size > 1:
            if "gpu" in self.arg.use_gpu_npu:
                dist.init_process_group(
                    backend='nccl',
                    init_method='env://',
                    world_size=self.world_size,
                    rank=self.rank
                )
            elif "npu" in self.arg.use_gpu_npu:
                dist.init_process_group(
                    backend='hccl',
                    world_size=self.world_size,
                    rank=self.rank
                )
        self.load_model()
        self.model = self.model.to(self.dev)
        self.mv_to_device()
        self.load_weights()
        self.load_optimizer()
        if self.amp:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer,
                                                        opt_level='O2',
                                                        keep_batchnorm_fp32=None,
                                                        loss_scale=64
                                                        )
        if self.world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.rank])
        self.load_data()
        self.best_acc = 0
        self.start()

    def init_environment(self):

        super().init_environment()
        self.result = dict()
        self.iter_info = dict()
        self.epoch_info = dict()
        self.meta_info = dict(epoch=0, iter=0)

    def load_optimizer(self):
        pass

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        if 'debug' not in self.arg.train_feeder_args:
            self.arg.train_feeder_args['debug'] = self.arg.debug
        self.data_loader = dict()
        if self.arg.phase == 'train':
            train_dataset = Feeder(**self.arg.train_feeder_args)
            if self.world_size > 1:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                    train_dataset)
            else:
                self.train_sampler = None
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=self.arg.batch_size,
                shuffle=(self.train_sampler is None),
                num_workers=self.arg.num_worker,
                pin_memory=True,
                drop_last=True,
                sampler=self.train_sampler)
        if self.arg.test_feeder_args:
            self.data_loader['test'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.test_feeder_args),
                batch_size=self.arg.test_batch_size,
                shuffle=False,
                num_workers=self.arg.num_worker)

    def show_epoch_info(self):
        for k, v in self.epoch_info.items():
            self.io.print_log('\t{}: {}'.format(k, v))
        if self.arg.pavi_log:
            self.io.log('train', self.meta_info['iter'], self.epoch_info)

    def show_iter_info(self):
        if self.meta_info['iter'] % self.arg.log_interval == 0:
            info = '\tIter {} Done.'.format(self.meta_info['iter'])
            for k, v in self.iter_info.items():
                if isinstance(v, float):
                    info = info + ' | {}: {:.4f}'.format(k, v)
                else:
                    info = info + ' | {}: {}'.format(k, v)

            self.io.print_log(info)

            if self.arg.pavi_log:
                self.io.log('train', self.meta_info['iter'], self.iter_info)

    def train(self):
        for _ in range(100):
            self.iter_info['loss'] = 0
            self.show_iter_info()
            self.meta_info['iter'] += 1
        self.epoch_info['mean loss'] = 0
        self.show_epoch_info()

    def test(self):
        for _ in range(100):
            self.iter_info['loss'] = 1
            self.show_iter_info()
        self.epoch_info['mean loss'] = 1
        self.show_epoch_info()

    def start(self):
        self.io.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))

        # training phase
        if self.arg.phase == 'train':
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                self.meta_info['epoch'] = epoch

                # training
                self.io.print_log('Training epoch: {}'.format(epoch))
                if self.world_size > 1:
                    self.train_sampler.set_epoch(epoch)
                self.train()
                self.io.print_log('Done.')
                self.meta_info['iter'] = 0

                # save model
                if ((epoch + 1) % self.arg.save_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch):
                    if self.rank == 0 or self.world_size == 1:
                        filename = 'epoch{}_model_{}p.pt'.format(
                            epoch + 1, self.world_size)
                        self.io.save_model(self.model, filename)

                # evaluation
                if ((epoch + 1) % self.arg.eval_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch):
                    self.io.print_log('Eval epoch: {}'.format(epoch))
                    self.test()
                    self.io.print_log('Done.')
        # test phase
        elif self.arg.phase == 'test':

            # the path of weights must be appointed
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.io.print_log('Model:   {}.'.format(self.arg.model))
            self.io.print_log('Weights: {}.'.format(self.arg.weights))

            # evaluation
            self.io.print_log('Evaluation Start:')
            self.test()
            self.io.print_log('Done.\n')

            # save the output of model
            if self.arg.save_result:
                result_dict = dict(
                    zip(self.data_loader['test'].dataset.sample_name,
                        self.result))
                self.io.save_pkl(result_dict, 'test_result.pkl')

    @staticmethod
    def get_parser(add_help=False):

        # region arguments yapf: disable
        # parameter priority: command line > config > default
        parser = argparse.ArgumentParser(
            add_help=add_help, description='Base Processor')

        parser.add_argument('-w', '--work_dir', default='./work_dir/tmp',
                            help='the work folder for storing results')
        parser.add_argument('-c', '--config', default=None,
                            help='path to the configuration file')

        # processor
        parser.add_argument('--phase', default='train',
                            help='must be train or test')
        parser.add_argument('--save_result', type=str2bool, default=False,
                            help='if ture, the output of the model will be stored')
        parser.add_argument('--start_epoch', type=int,
                            default=0, help='start training from which epoch')
        parser.add_argument('--num_epoch', type=int, default=80,
                            help='stop training in which epoch')
        parser.add_argument('--steps_per_epoch', type=int, default=3700,
                            help='steps to run in one epoch')
        parser.add_argument('--use_gpu_npu', type=str,
                            default="gpu", help='use GPU or NPU')
        parser.add_argument('--device', type=int, default=0, nargs='+',
                            help='the indexes of GPUs for training or testing')
        parser.add_argument('--amp', type=str2bool,
                            default=True, help='use amp or not')
        # visulize and debug
        parser.add_argument('--log_interval', type=int, default=100,
                            help='the interval for printing messages (#iteration)')
        parser.add_argument('--save_interval', type=int, default=10,
                            help='the interval for storing models (#iteration)')
        parser.add_argument('--eval_interval', type=int, default=1,
                            help='the interval for evaluating models (#iteration)')
        parser.add_argument('--save_log', type=str2bool,
                            default=True, help='save logging or not')
        parser.add_argument('--print_log', type=str2bool,
                            default=True, help='print logging or not')
        parser.add_argument('--pavi_log', type=str2bool,
                            default=False, help='logging on pavi or not')

        # feeder
        parser.add_argument('--feeder', default='feeder.feeder',
                            help='data loader will be used')
        parser.add_argument('--num_worker', type=int, default=4,
                            help='the number of worker per gpu for data loader')
        parser.add_argument('--train_feeder_args', action=DictAction,
                            default=dict(), help='the arguments of data loader for training')
        parser.add_argument('--test_feeder_args', action=DictAction,
                            default=dict(), help='the arguments of data loader for test')
        parser.add_argument('--batch_size', type=int,
                            default=256, help='training batch size')
        parser.add_argument('--test_batch_size', type=int,
                            default=256, help='test batch size')
        parser.add_argument('--debug', action="store_true",
                            help='less data, faster loading')

        # model
        parser.add_argument('--model', default=None,
                            help='the model will be used')
        parser.add_argument('--model_args', action=DictAction,
                            default=dict(), help='the arguments of model')
        parser.add_argument('--weights', default=None,
                            help='the weights for network initialization')
        parser.add_argument('--ignore_weights', type=str, default=[], nargs='+',
                            help='the name of weights which will be ignored in the initialization')
        # endregion yapf: enable

        # enable op binary
        parser.add_argument('--bin', type=str2bool, default=False, help='enable op binary')

        #profiling
        parser.add_argument('--profiling', type=str, default='NONE', help='choose profiling way --CANN, GE, NONE')
        parser.add_argument('--start_step', type=int, default=0, help='start step for profiling')
        parser.add_argument('--stop_step', type=int, default=20, help='stop step for profiling')

        return parser
