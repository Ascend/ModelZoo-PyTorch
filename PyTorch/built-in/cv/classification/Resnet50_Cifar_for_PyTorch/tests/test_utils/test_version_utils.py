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
from mmcls import digit_version


def test_digit_version():
    assert digit_version('0.2.16') == (0, 2, 16, 0, 0, 0)
    assert digit_version('1.2.3') == (1, 2, 3, 0, 0, 0)
    assert digit_version('1.2.3rc0') == (1, 2, 3, 0, -1, 0)
    assert digit_version('1.2.3rc1') == (1, 2, 3, 0, -1, 1)
    assert digit_version('1.0rc0') == (1, 0, 0, 0, -1, 0)
    assert digit_version('1.0') == digit_version('1.0.0')
    assert digit_version('1.5.0+cuda90_cudnn7.6.3_lms') == digit_version('1.5')
    assert digit_version('1.0.0dev') < digit_version('1.0.0a')
    assert digit_version('1.0.0a') < digit_version('1.0.0a1')
    assert digit_version('1.0.0a') < digit_version('1.0.0b')
    assert digit_version('1.0.0b') < digit_version('1.0.0rc')
    assert digit_version('1.0.0rc1') < digit_version('1.0.0')
    assert digit_version('1.0.0') < digit_version('1.0.0post')
    assert digit_version('1.0.0post') < digit_version('1.0.0post1')
    assert digit_version('v1') == (1, 0, 0, 0, 0, 0)
    assert digit_version('v1.1.5') == (1, 1, 5, 0, 0, 0)
