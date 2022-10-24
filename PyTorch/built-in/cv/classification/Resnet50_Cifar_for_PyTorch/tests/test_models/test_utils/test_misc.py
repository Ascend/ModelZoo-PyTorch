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
import pytest
import torch
from mmcv.utils import digit_version

from mmcls.models.utils import channel_shuffle, is_tracing, make_divisible


def test_make_divisible():
    # test min_value is None
    result = make_divisible(34, 8, None)
    assert result == 32

    # test when new_value > min_ratio * value
    result = make_divisible(10, 8, min_ratio=0.9)
    assert result == 16

    # test min_value = 0.8
    result = make_divisible(33, 8, min_ratio=0.8)
    assert result == 32


def test_channel_shuffle():
    x = torch.randn(1, 24, 56, 56)
    with pytest.raises(AssertionError):
        # num_channels should be divisible by groups
        channel_shuffle(x, 7)

    groups = 3
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    out = channel_shuffle(x, groups)
    # test the output value when groups = 3
    for b in range(batch_size):
        for c in range(num_channels):
            c_out = c % channels_per_group * groups + c // channels_per_group
            for i in range(height):
                for j in range(width):
                    assert x[b, c, i, j] == out[b, c_out, i, j]


@pytest.mark.skipif(
    digit_version(torch.__version__) < digit_version('1.6.0'),
    reason='torch.jit.is_tracing is not available before 1.6.0')
def test_is_tracing():

    def foo(x):
        if is_tracing():
            return x
        else:
            return x.tolist()

    x = torch.rand(3)
    # test without trace
    assert isinstance(foo(x), list)

    # test with trace
    traced_foo = torch.jit.trace(foo, (torch.rand(1), ))
    assert isinstance(traced_foo(x), torch.Tensor)
