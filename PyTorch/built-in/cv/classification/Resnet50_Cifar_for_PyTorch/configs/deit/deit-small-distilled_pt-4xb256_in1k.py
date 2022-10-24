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
_base_ = './deit-small_pt-4xb256_in1k.py'

# model settings
model = dict(
    backbone=dict(type='DistilledVisionTransformer', arch='deit-small'),
    head=dict(type='DeiTClsHead', in_channels=384),
)
