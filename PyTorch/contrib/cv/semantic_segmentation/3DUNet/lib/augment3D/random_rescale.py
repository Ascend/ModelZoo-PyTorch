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
import scipy.ndimage as ndimage


def random_zoom(img_numpy, min_percentage=0.8, max_percentage=1.1):
    """
    :param img_numpy: 
    :param min_percentage: 
    :param max_percentage: 
    :return: zoom in/out aigmented img
    """
    z = np.random.sample() * (max_percentage - min_percentage) + min_percentage
    zoom_matrix = np.array([[z, 0, 0, 0],
                            [0, z, 0, 0],
                            [0, 0, z, 0],
                            [0, 0, 0, 1]])
    return ndimage.interpolation.affine_transform(img_numpy, zoom_matrix)


class RandomZoom(object):
    def __init__(self, min_percentage=0.8, max_percentage=1.1):
        self.min_percentage = min_percentage
        self.max_percentage = max_percentage

    def __call__(self, img_numpy, label=None):
        img_numpy = random_zoom(img_numpy, self.min_percentage, self.max_percentage)
        if label.any() != None:
            label = random_zoom(label, self.min_percentage, self.max_percentage)
        return img_numpy, label
