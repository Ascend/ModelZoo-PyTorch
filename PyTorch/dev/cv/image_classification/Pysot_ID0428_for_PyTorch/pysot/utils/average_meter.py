#
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
#

# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


class Meter(object):
    def __init__(self, name, val, avg):
        self.name = name
        self.val = val
        self.avg = avg

    def __repr__(self):
        return "{name}: {val:.6f} ({avg:.6f})".format(
            name=self.name, val=self.val, avg=self.avg
        )

    def __format__(self, *tuples, **kwargs):
        return self.__repr__()


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self, num=100):
        self.num = num
        self.reset()

    def reset(self):
        self.val = {}
        self.sum = {}
        self.count = {}
        self.history = {}

    def update(self, batch=1, **kwargs):
        val = {}
        for k in kwargs:
            val[k] = kwargs[k] / float(batch)
        self.val.update(val)
        for k in kwargs:
            if k not in self.sum:
                self.sum[k] = 0
                self.count[k] = 0
                self.history[k] = []
            self.sum[k] += kwargs[k]
            self.count[k] += batch
            for _ in range(batch):
                self.history[k].append(val[k])

            if self.num <= 0:
                # < 0, average all
                self.history[k] = []

                # == 0: no average
                if self.num == 0:
                    self.sum[k] = self.val[k]
                    self.count[k] = 1

            elif len(self.history[k]) > self.num:
                pop_num = len(self.history[k]) - self.num
                for _ in range(pop_num):
                    self.sum[k] -= self.history[k][0]
                    del self.history[k][0]
                    self.count[k] -= 1

    def __repr__(self):
        s = ''
        for k in self.sum:
            s += self.format_str(k)
        return s

    def format_str(self, attr):
        return "{name}: {val:.6f} ({avg:.6f}) ".format(
                    name=attr,
                    val=float(self.val[attr]),
                    avg=float(self.sum[attr]) / self.count[attr])

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return super(AverageMeter, self).__getattr__(attr)
        if attr not in self.sum:
            print("invalid key '{}'".format(attr))
            return Meter(attr, 0, 0)
        return Meter(attr, self.val[attr], self.avg(attr))

    def avg(self, attr):
        return float(self.sum[attr]) / self.count[attr]


if __name__ == '__main__':
    avg1 = AverageMeter(10)
    avg2 = AverageMeter(0)
    avg3 = AverageMeter(-1)

    for i in range(20):
        avg1.update(s=i)
        avg2.update(s=i)
        avg3.update(s=i)

        print('iter {}'.format(i))
        print(avg1.s)
        print(avg2.s)
        print(avg3.s)
