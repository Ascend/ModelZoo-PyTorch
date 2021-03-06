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

import numpy as np
from numpy.testing import assert_array_almost_equal

from mmpose.core import compute_similarity_transform


def test_compute_similarity_transform():
    source = np.random.rand(14, 3)
    tran = np.random.rand(1, 3)
    scale = 0.5
    target = source * scale + tran
    source_transformed = compute_similarity_transform(source, target)
    assert_array_almost_equal(source_transformed, target)
