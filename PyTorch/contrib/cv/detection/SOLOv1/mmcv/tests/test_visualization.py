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

# Copyright (c) Open-MMLab. All rights reserved.
import numpy as np
import pytest

import mmcv


def test_color():
    assert mmcv.color_val(mmcv.Color.blue) == (255, 0, 0)
    assert mmcv.color_val('green') == (0, 255, 0)
    assert mmcv.color_val((1, 2, 3)) == (1, 2, 3)
    assert mmcv.color_val(100) == (100, 100, 100)
    assert mmcv.color_val(np.zeros(3, dtype=np.int)) == (0, 0, 0)
    with pytest.raises(TypeError):
        mmcv.color_val([255, 255, 255])
    with pytest.raises(TypeError):
        mmcv.color_val(1.0)
    with pytest.raises(AssertionError):
        mmcv.color_val((0, 0, 500))
