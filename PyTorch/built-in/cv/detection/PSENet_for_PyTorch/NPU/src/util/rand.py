# coding=utf-8
'''
Created on 2016年9月27日

@author: dengdan
'''
#     Copyright [yyyy] [name of copyright owner]
#     Copyright 2020 Huawei Technologies Co., Ltd
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
#



import random
import time

import numpy as np

rng = np.random.RandomState(int(time.time()))

rand = np.random.rand
"""
Create an array of the given shape and populate it with random samples from a uniform distribution over [0, 1)
"""


def normal(shape, mu=0, sigma_square=1):
    return rng.normal(mu, np.sqrt(sigma_square), shape)


def randint(low=2 ** 30, high=None, shape=None):
    """
    low: the higher bound except when high is not None.
    high: when it is not none, low must be smaller than it
    shape: if not provided, a scalar will be returned
    """
    return rng.randint(low=low, high=high, size=shape)


def shuffle(lst):
    random.shuffle(lst)


def sample(lst, n):
    return random.sample(lst, n)
