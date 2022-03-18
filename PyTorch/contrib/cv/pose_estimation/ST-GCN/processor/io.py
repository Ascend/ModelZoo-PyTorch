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

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class


class IO():
    """
        IO Processor
    """

    def __init__(self, argv=None):

        self.load_arg(argv)
        self.init_environment()
        self.load_model()
        self.load_weights()
        self.mv_to_device()

    def load_arg(self, argv=None):
        parser = self.get_parser()

        # load arg form config file
        p = parser.parse_args(argv)
        if p.config is not None:
            # load config file
            with open(p.config, 'r') as f:
                default_arg = yaml.load(f, Loader=yaml.FullLoader)

            # update parser from config file
            key = vars(p).keys()
            for k in default_arg.keys():
                if k not in key:
                    print('Unknown Arguments: {}'.format(k))
                    assert k in key

            parser.set_defaults(**default_arg)

        self.arg = parser.parse_args(argv)

    def init_environment(self):
        self.io = torchlight.IO(
            self.arg.work_dir,
            save_log=self.arg.save_log,
            print_log=self.arg.print_log)
        if self.rank == 0 or self.world_size == 1:
            self.io.save_arg(self.arg)

        # gpu
        if "gpu" in self.arg.use_gpu_npu:
            print("Using Gpu")
            gpus = torchlight.visible_gpu(self.arg.device)
            torchlight.occupy_gpu(gpus)
            self.dev = 'cuda:{}'.format(self.rank)
            torch.cuda.set_device(self.dev)
        elif "npu" in self.arg.use_gpu_npu:
            print("Using Npu")
            self.dev = 'npu:{}'.format(self.rank)
            torch.npu.set_device(self.dev)
        else:
            self.dev = "cpu"

    def load_model(self):
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))

    def load_weights(self):
        if self.arg.weights:
            self.model = self.io.load_weights(self.model, self.arg.weights,
                                              self.arg.ignore_weights)

    def mv_to_device(self):
        # move modules to device
        for name, value in vars(self).items():
            cls_name = str(value.__class__)
            if cls_name.find('torch.nn.modules') != -1:
                setattr(self, name, value.to(self.dev))

    def start(self):
        self.io.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))

    @staticmethod
    def get_parser(add_help=False):

        # region arguments yapf: disable
        # parameter priority: command line > config > default
        parser = argparse.ArgumentParser(
            add_help=add_help, description='IO Processor')

        parser.add_argument('-w', '--work_dir', default='./work_dir/tmp',
                            help='the work folder for storing results')
        parser.add_argument('-c', '--config', default=None,
                            help='path to the configuration file')

        # processor
        parser.add_argument('--use_gpu_npu', type=str,
                            default="gpu", help='use GPU or NPU')
        parser.add_argument('--device', type=int, default=0, nargs='+',
                            help='the indexes of GPUs for training or testing')

        # visulize and debug
        parser.add_argument('--print_log', type=str2bool,
                            default=True, help='print logging or not')
        parser.add_argument('--save_log', type=str2bool,
                            default=True, help='save logging or not')

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

        return parser
