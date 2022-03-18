# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright 2020 Huawei Technologies Co., Ltd
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

import unittest
import torch

from detectron2.structures.keypoints import Keypoints


class TestKeypoints(unittest.TestCase):
    def test_cat_keypoints(self):
        keypoints1 = Keypoints(torch.rand(2, 21, 3))
        keypoints2 = Keypoints(torch.rand(4, 21, 3))

        cat_keypoints = keypoints1.cat([keypoints1, keypoints2])
        self.assertTrue(torch.all(cat_keypoints.tensor[:2] == keypoints1.tensor).item())
        self.assertTrue(torch.all(cat_keypoints.tensor[2:] == keypoints2.tensor).item())


if __name__ == "__main__":
    unittest.main()
