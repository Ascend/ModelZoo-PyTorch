# -*- coding: utf-8 -*-
# Copyright 2021 Huawei Technologies Co., Ltd
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

import time
import sys
import os


class global_var:
    root_path = '../results/'


def get_root_path():
    return global_var.root_path


def set_root_path(path):
    if not path.endswith('/'):
        path = path + '/'
    global_var.root_path = path


class AverageMeter():
    '''用于统计程序中各个部分的运行时间，以及控制程序的中断点。'''

    def __init__(self, performance=False):
        self.performance = performance
        self.task_start = time.time()
        self._print_start_time(self.task_start)
        self.t_training = 0
        self.t_val = 0
        self.t_epoch = 0
        self.step = 0

    def t_start(self, name):
        if name == 'training':
            self.t_training = time.time()
        elif name == 'val':
            self.t_val = time.time()
        elif name == 'epoch':
            self.t_epoch = time.time()

    def _print_start_time(self, val):
        time_array = time.localtime(val)
        time_style = time.strftime("%Y-%m-%d %H:%M:%S", time_array)
        print('Task start: ', time_style)
        return

    def time_gap(self, start, end):
        gap_h = (end - start) // 3600
        gap_m = ((end - start) % 3600) // 60
        gap_s = ((end - start) % 3600) % 60
        gap_time = "%02d:%02d:%02d" % (gap_h, gap_m, gap_s)
        return gap_time

    def print_time(self, name):
        end = time.time()
        if name == 'training':
            gap_t = self.time_gap(self.t_training, end)
            print('Training one epoch time: ' + gap_t)
        elif name == 'val':
            gap_t = self.time_gap(self.t_val, end)
            print('Val time: ' + gap_t)
        elif name == 'epoch':
            gap_t = self.time_gap(self.t_epoch, end)
            print('Epoch time: ' + gap_t)
        elif name == 'end':
            gap_t = self.time_gap(self.task_start, end)
            print(f'{self.step} steps have been run.')
            print('The total elapsed time: ' + gap_t)

    def _total_time(self):
        end = time.time()
        gap_m = ((end - self.task_start) % 3600) // 60
        return gap_m

    def step_update(self):
        if self.performance:
            if self.step < 1001 and self._total_time() < 60:
                self.step += 1
            else:
                print(f'1000 steps have been run, The elapsed time is {self._total_time()} minutes ')
                sys.exit()
        else:
            self.step += 1

