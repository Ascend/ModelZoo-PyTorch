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
_base_ = [
    '../_base_/models/resnet50.py', '../_base_/datasets/cub_bs8_448.py',
    '../_base_/schedules/cub_bs64.py', '../_base_/default_runtime.py'
]

# use pre-train weight converted from https://github.com/Alibaba-MIIL/ImageNet21K # noqa
checkpoint = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_3rdparty-mill_in21k_20220331-faac000b.pth'  # noqa

model = dict(
    type='ImageClassifier',
    backbone=dict(
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint, prefix='backbone')),
    head=dict(num_classes=200, ))

log_config = dict(interval=20)  # log every 20 intervals

checkpoint_config = dict(
    interval=1, max_keep_ckpts=3)  # save last three checkpoints
