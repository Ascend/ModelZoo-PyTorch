
# Copyright 2022 Huawei Technologies Co., Ltd
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

# Copyright (c) Open-MMLab. All rights reserved.    
import os
from pathlib import Path

import pytest

from mmdet.apis import init_detector


def test_init_detector():
    project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    project_dir = os.path.join(project_dir, '..')

    config_file = os.path.join(
        project_dir, 'configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py')

    # test init_detector with config_file: str and cfg_options
    cfg_options = dict(
        model=dict(
            backbone=dict(
                depth=18,
                init_cfg=dict(
                    type='Pretrained', checkpoint='torchvision://resnet18'))))
    model = init_detector(config_file, device='cpu', cfg_options=cfg_options)

    # test init_detector with :obj:`Path`
    config_path_object = Path(config_file)
    model = init_detector(config_path_object, device='cpu')

    # test init_detector with undesirable type
    with pytest.raises(TypeError):
        config_list = [config_file]
        model = init_detector(config_list)  # noqa: F841
