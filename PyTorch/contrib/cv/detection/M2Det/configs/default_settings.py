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

# Only m2det_configs are supported
# For debug when configuring m2det.py file
# you can modify them in m2det_config in ./configs/m2detxxx.py

m2det_configs = dict(
    vgg16 = dict(
        backbone = 'vgg16',
        net_family = 'vgg',
        base_out = [22,34], # [22,34] for vgg, [2,4] or [3,4] for res families
        planes = 256,
        num_levels = 8,
        num_scales = 6,
        sfam = True,
        smooth = True,
        num_classes = 81,
        ),
    resnet50 = dict(
        backbone = 'resnet50',
        net_family = 'res',
        base_out = [2,4], # [22,34] for vgg, [2,4] or [3,4] for res families
        planes = 256,
        num_levels = 8,
        num_scales = 6,
        sfam = True,
        smooth = True,
        num_classes = 81,
        )
    )

