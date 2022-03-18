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
# ============================================================================
'''
为了计算训练FPS而添加的hooks
ckx 21/7/22
'''
import os.path as osp
import time
from mmcv.runner import Hook, master_only
 

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', start_count_index=10):
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
            self.avg = self.sum / (self.count - self.start_count_index * self.N)

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    

class EvalTrainHook(Hook):
    def __init__(self,batch_size, num_workers):
        self.iter_time  = AverageMeter('iterTime', ':6.3f', 5)
        self.batch_size = batch_size
        self.num_worker = num_workers
        
    def before_epoch(self, runner):
        runner.logger.info(f'new epoch start !')
        self.iter_time.reset()
    
    # 优先级最低
    def before_train_iter(self, runner):
        self.end = time.time()
    
    # 优先级, 在optimizer.step() 计算后,在 chenkpoint save前,
    def after_train_iter(self, runner):
        self.iter_time.update(time.time() - self.end)
        
    def after_epoch(self, runner):
        self._cal_fps(runner)

    # 只允许master节点计算和唤醒
    @master_only
    def _cal_fps(self, runner):
        res = self.batch_size / self.iter_time.avg if self.iter_time.avg else 0
        # print(f'epoch time avg = {self.iter_time.avg}')
        # print(f"train FPS = {res}")
        runner.logger.info(f'epoch_num: {runner.epoch}')
        runner.logger.info(f'epoch_time avg {self.iter_time.avg}')
        runner.logger.info(f'train_FPS: {res}')


