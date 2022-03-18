# This file is for backward compatibility.
# Module wrappers for empty tensor have been moved to mmcv.cnn.bricks.
# modify from https://github.com/rosinality/stylegan2-pytorch/blob/master/op/fused_act.py # noqa:E501
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import warnings

from ..cnn.bricks.wrappers import Conv2d, ConvTranspose2d, Linear, MaxPool2d


class Conv2d_deprecated(Conv2d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(
            'Importing Conv2d wrapper from "mmcv.ops" will be deprecated in'
            ' the future. Please import them from "mmcv.cnn" instead')


class ConvTranspose2d_deprecated(ConvTranspose2d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(
            'Importing ConvTranspose2d wrapper from "mmcv.ops" will be '
            'deprecated in the future. Please import them from "mmcv.cnn" '
            'instead')


class MaxPool2d_deprecated(MaxPool2d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(
            'Importing MaxPool2d wrapper from "mmcv.ops" will be deprecated in'
            ' the future. Please import them from "mmcv.cnn" instead')


class Linear_deprecated(Linear):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(
            'Importing Linear wrapper from "mmcv.ops" will be deprecated in'
            ' the future. Please import them from "mmcv.cnn" instead')
