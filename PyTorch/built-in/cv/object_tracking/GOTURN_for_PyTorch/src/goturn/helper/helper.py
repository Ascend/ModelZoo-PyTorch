
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


import math

import torch

RAND_MAX = 2147483647


def sample_rand_uniform():
    """TODO: Docstring for sample_rand_uniform.

    :arg1: TODO
    :returns: TODO

    """
    # return ((random.randint(0, RAND_MAX) + 1) * 1.0) / (RAND_MAX + 2)
    rand_num = torch.randint(RAND_MAX, (1, 1)).item()
    return ((rand_num + 1) / (RAND_MAX + 2))
    # return torch.rand(1).item()


def sample_exp_two_sides(lambda_):
    """TODO: Docstring for sample_exp_two_sides.
    :returns: TODO

    """

    # pos_or_neg = random.randint(0, RAND_MAX)
    pos_or_neg = torch.randint(RAND_MAX, (1, 1)).item()
    if (pos_or_neg % 2) == 0:
        pos_or_neg = 1
    else:
        pos_or_neg = -1

    rand_uniform = sample_rand_uniform()
    return math.log(rand_uniform) / (lambda_ * pos_or_neg)


if __name__ == "__main__":
    # out = sample_rand_uniform()
    # torch.manual_seed(800)
    # out = torch.rand(1)
    for i in range(1000000000000000000):
        print(sample_exp_two_sides(0.4))
