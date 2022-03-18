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

"""A wrapper class for optimizer"""
import torch


class TransformerOptimizer(object):
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer, k, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.k = k
        self.init_lr = d_model ** (-0.5)
        self.warmup_steps = warmup_steps
        self.step_num = 0
        self.visdom_lr = None

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self._update_lr()
        self._visdom()
        self.optimizer.step()

    def _update_lr(self):
        self.step_num += 1
        lr = self.k * self.init_lr * min(self.step_num ** (-0.5),
                                         self.step_num * (self.warmup_steps ** (-1.5)))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def state_dict(self):
        return self.optimizer.state_dict()

    def set_k(self, k):
        self.k = k

    def set_visdom(self, visdom_lr, vis):
        self.visdom_lr = visdom_lr  # Turn on/off visdom of learning rate
        self.vis = vis  # visdom enviroment
        self.vis_opts = dict(title='Learning Rate',
                             ylabel='Leanring Rate', xlabel='step')
        self.vis_window = None
        self.x_axis = torch.LongTensor()
        self.y_axis = torch.FloatTensor()

    def _visdom(self):
        if self.visdom_lr is not None:
            self.x_axis = torch.cat(
                [self.x_axis, torch.LongTensor([self.step_num])])
            self.y_axis = torch.cat(
                [self.y_axis, torch.FloatTensor([self.optimizer.param_groups[0]['lr']])])
            if self.vis_window is None:
                self.vis_window = self.vis.line(X=self.x_axis, Y=self.y_axis,
                                                opts=self.vis_opts)
            else:
                self.vis.line(X=self.x_axis, Y=self.y_axis, win=self.vis_window,
                              update='replace')
