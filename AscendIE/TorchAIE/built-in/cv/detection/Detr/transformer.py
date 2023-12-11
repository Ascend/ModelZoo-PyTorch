# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import random
import torchvision.transforms.functional as F
import numpy as np


def resize(image, size, max_size=None):
    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))
        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)
        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    h, w = size[0], size[1]
    scale = np.array([i for i in range(768, 1400, 256)])
    min_scale = np.array([i for i in range(512, 900, 256)])

    def resize_step(scale, x, step):
        value = min(abs(scale - x))
        if (x + value) % step == 0:
            x = x + value
            return x
        elif (x - value) % step == 0:
            x = x - value
            return x

    if h == 768:
        size = (h, resize_step(scale, w, step=256))
    elif w == 768:
        size = (resize_step(scale, h, step=256), w)
    elif h < 768:
        size = (resize_step(min_scale, h, step=256), 1344)
    elif w < 768:
        size = (1344, resize_step(min_scale, w, step=256))
    rescaled_image = F.resize(image, size)
    return rescaled_image


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img):
        size = random.choice(self.sizes)
        return resize(img, size, self.max_size)


class ToTensor(object):
    def __call__(self, img):
        return F.to_tensor(img)


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string