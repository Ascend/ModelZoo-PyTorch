# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright (c) Open-MMLab. All rights reserved.
import numbers

from ...dist_utils import master_only
from .base import LoggerHook


class WandbLoggerHook(LoggerHook):

    def __init__(self,
                 log_dir=None,
                 interval=10,
                 ignore_last=True,
                 reset_flag=True):
        super(WandbLoggerHook, self).__init__(interval, ignore_last,
                                              reset_flag)
        self.import_wandb()

    def import_wandb(self):
        try:
            import wandb
        except ImportError:
            raise ImportError(
                'Please run "pip install wandb" to install wandb')
        self.wandb = wandb

    @master_only
    def before_run(self, runner):
        if self.wandb is None:
            self.import_wandb()
        self.wandb.init()

    @master_only
    def log(self, runner):
        metrics = {}
        for var, val in runner.log_buffer.output.items():
            if var in ['time', 'data_time']:
                continue
            tag = '{}/{}'.format(var, runner.mode)
            runner.log_buffer.output[var]
            if isinstance(val, numbers.Number):
                metrics[tag] = val
        if metrics:
            self.wandb.log(metrics, step=runner.iter)

    @master_only
    def after_run(self, runner):
        self.wandb.join()
