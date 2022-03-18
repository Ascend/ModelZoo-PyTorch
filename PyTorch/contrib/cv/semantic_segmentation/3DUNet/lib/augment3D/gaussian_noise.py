# Copyright 2021 Huawei Technologies Co., Ltd
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


import numpy as np


def random_noise(img_numpy, mean=0, std=0.001):
    noise = np.random.normal(mean, std, img_numpy.shape)

    return img_numpy + noise


class GaussianNoise(object):
    def __init__(self, mean=0, std=0.001):
        self.mean = mean
        self.std = std

    def __call__(self, img_numpy, label=None):
        """
        Args:
            img_numpy (numpy): Image to be flipped.
            label (numpy): Label segmentation map to be flipped

        Returns:
            img_numpy (numpy):  flipped img.
            label (numpy): flipped Label segmentation.
        """

        return random_noise(img_numpy, self.mean, self.std), label
