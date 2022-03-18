#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
"""
pytorch-dl
Created by raj at 11:05
Date: January 18, 2020
"""

import numpy as np
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))


def sigmoid(x, derivative=False):
    if derivative:
        return x * (1 - x)
    else:
        return 1 / (1+np.exp(-x))


class Network(object):
    def __init__(self, x, y):
        self.input = x
        self.w1 = np.random.rand(self.input.shape[1], 4)
        self.w2 = np.random.rand(4, 4)
        self.w3 = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(self.y.shape)

    def feed_forward(self):
        self.y1 = sigmoid(np.dot(self.input, self.w1))
        self.y2 = sigmoid(np.dot(self.y1, self.w2))
        self.y3 = sigmoid(np.dot(self.y2, self.w3))
        return self.y3

    def backward(self):
        error_l3 = 2 * (self.y - self.output) * sigmoid(self.output, derivative=True)
        delta3 = np.dot(self.y2.T, error_l3)
        error_l2 = np.dot(error_l3, self.w3.T)*sigmoid(self.y2, derivative=True)
        delta2 = np.dot(self.y1.T, error_l2)
        error_l1 = np.dot(error_l2, self.w2.T) * sigmoid(self.y1, derivative=True)
        delta1 = np.dot(self.input.T, error_l1)

        self.w1 += delta1
        self.w2 += delta2
        self.w3 += delta3

    def train(self):
        self.output = self.feed_forward()
        self.backward()


x=np.array(([0,0,1],[0,1,1],[1,0,1],[1,1,1]), dtype=float)
y=np.array(([0],[1],[1],[0]), dtype=float)

NN = Network(x, y)
for i in range(1500):  # trains the NN 1,000 times
    if i % 100 == 0:
        print("for iteration # " + str(i) + "\n")
        print("Input : \n" + str(x))
        print("Actual Output: \n" + str(y))
        print("Predicted Output: \n" + str(NN.feed_forward()))
        print("Loss: \n" + str(np.mean(np.square(y - NN.feed_forward()))))  # mean sum squared loss
        print("\n")

    NN.train()