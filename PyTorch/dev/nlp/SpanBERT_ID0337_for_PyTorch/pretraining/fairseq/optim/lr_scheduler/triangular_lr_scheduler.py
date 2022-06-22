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
import math

from . import FairseqLRScheduler, register_lr_scheduler


@register_lr_scheduler('triangular')
class TriangularSchedule(FairseqLRScheduler):
    """Assign LR based on a triangular cyclical schedule.

    See https://arxiv.org/pdf/1506.01186.pdf for details

    """

    def __init__(self, args, optimizer):
        super().__init__(args, optimizer)
        if len(args.lr) > 1:
            raise ValueError(
                'Cannot use a fixed learning rate schedule with triangular.'
                ' Consider --lr-scheduler=fixed instead.'
            )

        lr = args.lr[0]

        assert args.max_lr > lr, 'max_lr must be more than lr'
        self.min_lr = lr
        self.max_lr = args.max_lr
        self.stepsize = args.lr_period_updates // 2
        self.lr_shrink = args.lr_shrink
        self.shrink_min = args.shrink_min

        # initial learning rate
        self.lr = self.min_lr
        self.optimizer.set_lr(self.lr)

    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        parser.add_argument('--max-lr', required=True, type=float, metavar='LR',
                            help='max learning rate, must be more than args.lr')
        parser.add_argument('--lr-period-updates', default=5000, type=float, metavar='LR',
                            help='initial number of updates per period (cycle length)')
        parser.add_argument('--shrink-min', action='store_true',
                            help='if set, also shrinks min lr')

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        super().step(epoch, val_loss)
        # we don't change the learning rate at epoch boundaries
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        cycle = math.floor(num_updates / (2 * self.stepsize))

        lr_shrink = self.lr_shrink ** cycle
        max_lr = self.max_lr * lr_shrink
        if self.shrink_min:
            min_lr = self.min_lr * lr_shrink
        else:
            min_lr = self.min_lr

        x = abs(num_updates / self.stepsize - 2 * (cycle + 1) + 1)
        self.lr = min_lr + (max_lr - min_lr) * max(0, (1 - x))

        self.optimizer.set_lr(self.lr)
        return self.lr
