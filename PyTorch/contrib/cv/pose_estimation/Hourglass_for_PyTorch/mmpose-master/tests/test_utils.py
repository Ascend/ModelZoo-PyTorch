# -*- coding: utf-8 -*-
# Copyright 2021 Huawei Technologies Co., Ltd
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
# ============================================================================

import cv2
import mmcv
import torch
import torchvision

import mmpose
from mmpose.utils import collect_env


def test_collect_env():
    env_info = collect_env()
    assert env_info['PyTorch'] == torch.__version__
    assert env_info['TorchVision'] == torchvision.__version__
    assert env_info['OpenCV'] == cv2.__version__
    assert env_info['MMCV'] == mmcv.__version__
    assert '+' in env_info['MMPose']
    assert mmpose.__version__ in env_info['MMPose']
