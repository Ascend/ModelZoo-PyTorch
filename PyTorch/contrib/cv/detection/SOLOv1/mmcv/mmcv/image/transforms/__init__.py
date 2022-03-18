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
from .colorspace import (bgr2gray, bgr2hls, bgr2hsv, bgr2rgb, gray2bgr,
                         gray2rgb, hls2bgr, hsv2bgr, iminvert, posterize,
                         rgb2bgr, rgb2gray, solarize)
from .geometry import imcrop, imflip, impad, impad_to_multiple, imrotate
from .normalize import imdenormalize, imnormalize
from .resize import imrescale, imresize, imresize_like

__all__ = [
    'solarize', 'posterize', 'bgr2gray', 'rgb2gray', 'gray2bgr', 'gray2rgb',
    'bgr2rgb', 'rgb2bgr', 'bgr2hsv', 'hsv2bgr', 'bgr2hls', 'hls2bgr',
    'iminvert', 'imflip', 'imrotate', 'imcrop', 'impad', 'impad_to_multiple',
    'imnormalize', 'imdenormalize', 'imresize', 'imresize_like', 'imrescale'
]
