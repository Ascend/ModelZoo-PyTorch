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
import time

from .hook import Hook


class IterTimerHook(Hook):

    def before_epoch(self, runner):
        self.t = time.time()
        self.skip_step = 0
        self.time_all = 0

    def before_iter(self, runner):
        runner.log_buffer.update({'data_time': time.time() - self.t})

    def after_iter(self, runner):
        ## npu diff
        # runner.log_buffer.update({'time': time.time() - self.t})
        cur_time = time.time()
        runner.log_buffer.update({'time': time.time() - self.t})
        if self.skip_step >= 5:
            self.time_all += cur_time - self.t
        self.skip_step += 1
        self.t = time.time()
