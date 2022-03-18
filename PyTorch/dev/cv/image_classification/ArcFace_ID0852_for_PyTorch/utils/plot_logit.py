#!/usr/bin/env python
# encoding: utf-8
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
'''
@author: wujiyang
@contact: wujiyang@hust.edu.cn
@file: plot_logit.py
@time: 2019/3/29 14:21
@desc: plot the logit corresponding to shpereface, cosface, arcface and so on.
'''

import math
import torch
import matplotlib.pyplot as plt
import numpy as np

def softmax(theta):
    return torch.cos(theta)

def sphereface(theta, m=4):
    return (torch.cos(m * theta) + 20 * torch.cos(theta)) / (20 + 1)

def cosface(theta, m):
    return torch.cos(theta) - m

def arcface(theta, m):
    return torch.cos(theta + m)

def multimargin(theta, m1, m2):
    return torch.cos(theta + m1) - m2


theta = torch.arange(0, math.pi, 0.001)
print(theta.type)

x = theta.numpy()
y_softmax = softmax(theta).numpy()
y_cosface = cosface(theta, 0.35).numpy()
y_arcface = arcface(theta, 0.5).numpy()

y_multimargin_1 = multimargin(theta, 0.2, 0.3).numpy()
y_multimargin_2 = multimargin(theta, 0.2, 0.4).numpy()
y_multimargin_3 = multimargin(theta, 0.3, 0.2).numpy()
y_multimargin_4 = multimargin(theta, 0.3, 0.3).numpy()
y_multimargin_5 = multimargin(theta, 0.4, 0.2).numpy()
y_multimargin_6 = multimargin(theta, 0.4, 0.3).numpy()

plt.plot(x, y_softmax, x, y_cosface, x, y_arcface, x, y_multimargin_1, x, y_multimargin_2, x, y_multimargin_3, x, y_multimargin_4, x, y_multimargin_5, x, y_multimargin_6)
plt.legend(['Softmax(0.00, 0.00)', 'CosFace(0.00, 0.35)', 'ArcFace(0.50, 0.00)', 'MultiMargin(0.20, 0.30)', 'MultiMargin(0.20, 0.40)', 'MultiMargin(0.30, 0.20)', 'MultiMargin(0.30, 0.30)', 'MultiMargin(0.40, 0.20)', 'MultiMargin(0.40, 0.30)'])
plt.grid(False)
plt.xlim((0, 3/4*math.pi))
plt.ylim((-1.2, 1.2))

plt.xticks(np.arange(0, 2.4, 0.3))
plt.yticks(np.arange(-1.2, 1.2, 0.2))
plt.xlabel('Angular between the Feature and Target Center (Radian: 0 - 3/4 Pi)')
plt.ylabel('Target Logit')

plt.savefig('target logits')