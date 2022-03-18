# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright (c) Open-MMLab. All rights reserved.
import numpy as np

from .colorspace import bgr2rgb, rgb2bgr


def imnormalize(img, mean, std, to_rgb=True):
    img = img.astype(np.float32)
    if to_rgb:
        img = bgr2rgb(img)
    return (img - mean) / std


def imdenormalize(img, mean, std, to_bgr=True):
    img = (img * std) + mean
    if to_bgr:
        img = rgb2bgr(img)
    return img
