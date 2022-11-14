# -*- coding: utf-8 -*-
# BSD 3-Clause License
#
# Copyright (c) 2017
# All rights reserved.
# Copyright 2022 Huawei Technologies Co., Ltd
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
# ==========================================================================

# Copyright (c) OpenMMLab. All rights reserved.
import inspect
import random

import mmcv
import numpy as np
import torchvision.transforms as torchvision_transforms
from mmcv.utils import build_from_cfg
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import Compose
from PIL import Image


@PIPELINES.register_module()
class OneOfWrapper:
    """Randomly select and apply one of the transforms, each with the equal
    chance.

    Warning:
        Different from albumentations, this wrapper only runs the selected
        transform, but doesn't guarantee the transform can always be applied to
        the input if the transform comes with a probability to run.

    Args:
        transforms (list[dict|callable]): Candidate transforms to be applied.
    """

    def __init__(self, transforms):
        assert isinstance(transforms, list) or isinstance(transforms, tuple)
        assert len(transforms) > 0, 'Need at least one transform.'
        self.transforms = []
        for t in transforms:
            if isinstance(t, dict):
                self.transforms.append(build_from_cfg(t, PIPELINES))
            elif callable(t):
                self.transforms.append(t)
            else:
                raise TypeError('transform must be callable or a dict')

    def __call__(self, results):
        return random.choice(self.transforms)(results)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(transforms={self.transforms})'
        return repr_str


@PIPELINES.register_module()
class RandomWrapper:
    """Run a transform or a sequence of transforms with probability p.

    Args:
        transforms (list[dict|callable]): Transform(s) to be applied.
        p (int|float): Probability of running transform(s).
    """

    def __init__(self, transforms, p):
        assert 0 <= p <= 1
        self.transforms = Compose(transforms)
        self.p = p

    def __call__(self, results):
        return results if np.random.uniform() > self.p else self.transforms(
            results)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(transforms={self.transforms}, '
        repr_str += f'p={self.p})'
        return repr_str


@PIPELINES.register_module()
class TorchVisionWrapper:
    """A wrapper of torchvision trasnforms. It applies specific transform to
    ``img`` and updates ``img_shape`` accordingly.

    Warning:
        This transform only affects the image but not its associated
        annotations, such as word bounding boxes and polygon masks. Therefore,
        it may only be applicable to text recognition tasks.

    Args:
        op (str): The name of any transform class in
            :func:`torchvision.transforms`.
        **kwargs: Arguments that will be passed to initializer of torchvision
            transform.

    :Required Keys:
        - | ``img`` (ndarray): The input image.

    :Affected Keys:
        :Modified:
            - | ``img`` (ndarray): The modified image.
        :Added:
            - | ``img_shape`` (tuple(int)): Size of the modified image.
    """

    def __init__(self, op, **kwargs):
        assert type(op) is str

        if mmcv.is_str(op):
            obj_cls = getattr(torchvision_transforms, op)
        elif inspect.isclass(op):
            obj_cls = op
        else:
            raise TypeError(
                f'type must be a str or valid type, but got {type(type)}')
        self.transform = obj_cls(**kwargs)
        self.kwargs = kwargs

    def __call__(self, results):
        assert 'img' in results
        # BGR -> RGB
        img = results['img'][..., ::-1]
        img = Image.fromarray(img)
        img = self.transform(img)
        img = np.asarray(img)
        img = img[..., ::-1]
        results['img'] = img
        results['img_shape'] = img.shape
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(transform={self.transform})'
        return repr_str
