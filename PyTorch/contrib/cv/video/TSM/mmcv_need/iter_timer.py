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
import time

from .hook import HOOKS, Hook


@HOOKS.register_module()
class IterTimerHook(Hook):

    def before_epoch(self, runner):
        self.t = time.time()

    def before_iter(self, runner):
        runner.log_buffer.update({'data_time': time.time() - self.t})

    def after_iter(self, runner):
        if runner.distributed:
            runner.log_buffer.update({'time': time.time() - self.t, 'fps': runner.batch_size * 8 / (time.time() - self.t)})
        else:
            runner.log_buffer.update({'time': time.time() - self.t, 'fps': runner.batch_size / (time.time() - self.t)})
        self.t = time.time()
