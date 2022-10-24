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
# optimizer
optimizer = dict(type='Lamb', lr=0.005, weight_decay=0.02)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1.0e-6,
    warmup='linear',
    # For ImageNet-1k, 626 iters per epoch, warmup 5 epochs.
    warmup_iters=5 * 626,
    warmup_ratio=0.0001)
runner = dict(type='EpochBasedRunner', max_epochs=100)
