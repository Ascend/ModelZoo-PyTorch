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
_base_ = 'resnet50_32xb64-warmup-lbs_in1k.py'

_deprecation_ = dict(
    expected='resnet50_32xb64-warmup-lbs_in1k.py',
    reference='https://github.com/open-mmlab/mmclassification/pull/508',
)
