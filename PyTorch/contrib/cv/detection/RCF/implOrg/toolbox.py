# Copyright (c) Soumith Chintala 2016,
# All rights reserved
#
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://spdx.org/licenses/BSD-3-Clause.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import numpy as np
from scipy import signal


def conv_tri(image, r, s=1):
    """ 2D image convolution with a triangle filter (no fast)
    See https://github.com/pdollar/toolbox/blob/master/channels/convTri.m
    Note: signal.convolve2d does not support float16('single' in MATLAB)
    """
    if image.size == 0 or (r == 0 and s == 1):
        return image
    if r <= 1:
        p = 12 / r / (r + 2) - 2
        f = np.array([[1, p, 1]]) / (2 + p)
        r = 1
    else:
        f = np.array([list(range(1, r + 1)) + [r + 1] + list(range(r, 0, -1))]) / (r + 1) ** 2
    f = f.astype(image.dtype)
    image = np.pad(image, ((r, r), (r, r)), mode="symmetric")
    image = signal.convolve2d(signal.convolve2d(image, f, "valid"), f.T, "valid")
    if s > 1:
        t = int(np.floor(s / 2) + 1)
        image = image[t-1:image.shape[0]-(s-t)+1:s, t-1:image.shape[1]-(s-t)+1:s]
    return image


def grad2(image):
    """ numerical gradients along x and y directions (no fast)
    See https://github.com/pdollar/toolbox/blob/master/channels/gradient2.m
    Note: np.gradient return [oy, ox], MATLAB version return [ox, oy]
    """
    assert image.ndim == 2
    oy, ox = np.gradient(image)
    return ox, oy


class Time:
    def __init__(self):
        self.time = None

    def set(self):
        self.time = time.time()

    def get(self):
        return time.time() - self.time


