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
from .losses import (CharbonnierLoss, GANLoss, L1Loss, MSELoss, PerceptualLoss,
                     WeightedTVLoss, g_path_regularize, compute_gradient_penalty,
                     r1_penalty)
import ascend_function

__all__ = [
    'L1Loss', 'MSELoss', 'CharbonnierLoss', 'WeightedTVLoss', 'PerceptualLoss',
    'GANLoss', 'compute_gradient_penalty', 'r1_penalty', 'g_path_regularize'
]
