# Copyright 2021 Huawei Technologies Co., Ltd
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


from mmdet import digit_version


def test_version_check():
    assert digit_version('1.0.5') > digit_version('1.0.5rc0')
    assert digit_version('1.0.5') > digit_version('1.0.4rc0')
    assert digit_version('1.0.5') > digit_version('1.0rc0')
    assert digit_version('1.0.0') > digit_version('0.6.2')
    assert digit_version('1.0.0') > digit_version('0.2.16')
    assert digit_version('1.0.5rc0') > digit_version('1.0.0rc0')
    assert digit_version('1.0.0rc1') > digit_version('1.0.0rc0')
    assert digit_version('1.0.0rc2') > digit_version('1.0.0rc0')
    assert digit_version('1.0.0rc2') > digit_version('1.0.0rc1')
    assert digit_version('1.0.1rc1') > digit_version('1.0.0rc1')
    assert digit_version('1.0.0') > digit_version('1.0.0rc1')
