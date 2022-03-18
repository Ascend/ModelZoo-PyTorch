# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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

import torch
import numpy as np

from .image_transforms import *

class ToTensor(Transform):
    """Converts a numpy.ndarray (H x W x (T x C)) in the range
    [0, 255] to a torch.FloatTensor of shape (C x T x H x W) in the range [0.0, 1.0].
    """
    def __init__(self, dim=3):
        self.dim = dim

    def __call__(self, clip, idx=None, copy_id=0):
        if isinstance(clip, np.ndarray):
            H, W, _ = clip.shape
            # handle numpy array
            clip = torch.from_numpy(clip.reshape((H,W,-1,self.dim)).transpose((3, 2, 0, 1)))
            # backward compatibility
            return clip.float() / 255.

