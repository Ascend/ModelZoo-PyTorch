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
from .builder import AUGMENT
from .utils import one_hot_encoding


@AUGMENT.register_module(name='Identity')
class Identity(object):
    """Change gt_label to one_hot encoding and keep img as the same.

    Args:
        num_classes (int): The number of classes.
        prob (float): MixUp probability. It should be in range [0, 1].
            Default to 1.0
    """

    def __init__(self, num_classes, prob=1.0):
        super(Identity, self).__init__()

        assert isinstance(num_classes, int)
        assert isinstance(prob, float) and 0.0 <= prob <= 1.0

        self.num_classes = num_classes
        self.prob = prob

    def one_hot(self, gt_label):
        return one_hot_encoding(gt_label, self.num_classes)

    def __call__(self, img, gt_label):
        return img, self.one_hot(gt_label)
