# encoding=utf-8
"""
for theano shortcuts
"""
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



import theano
import theano.tensor as T

import util.rand

trng = T.shared_randomstreams.RandomStreams(util.rand.randint())
scan_until = theano.scan_module.until


def add_noise(input, noise_level):
    noise = trng.binomial(size=input.shape, n=1, p=1 - noise_level)
    return noise * input


def crop_into(large, small):
    """
    center crop large image into small.
    both 'large' and 'small' are 4D: (batch_size, channels, h, w)
    """

    h1, w1 = large.shape[2:]
    h2, w2 = small.shape[2:]
    y, x = (h1 - h2) / 2, (w1 - h2) / 2
    return large[:, :, y: y + h2, x: x + w2]
