# Copyright (c) Open-MMLab. All rights reserved.
import time

from .hook import HOOKS, Hook


@HOOKS.register_module()
class IterTimerHook(Hook):

    def before_epoch(self, runner):
        self.t = time.time()
        self.skip_step = 0
        self.time_all = 0

    def before_iter(self, runner):
        runner.log_buffer.update({'data_time': time.time() - self.t})

    def after_iter(self, runner):
        # runner.log_buffer.update({'time': time.time() - self.t})
        cur_time = time.time()
        runner.log_buffer.update({'time': cur_time - self.t})
        if self.skip_step >= 5:
            self.time_all += cur_time - self.t
        self.skip_step += 1

        self.t = time.time()
