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

from detectron2.structures.masks import BitMasks, PolygonMasks, polygons_to_bitmask


class TestBitMask(unittest.TestCase):
    def test_get_bounding_box(self):
        masks = torch.tensor(
            [
                [
                    [False, False, False, True],
                    [False, False, True, True],
                    [False, True, True, False],
                    [False, True, True, False],
                ],
                [
                    [False, False, False, False],
                    [False, False, True, False],
                    [False, True, True, False],
                    [False, True, True, False],
                ],
                torch.zeros(4, 4),
            ]
        )
        bitmask = BitMasks(masks)
        box_true = torch.tensor([[1, 0, 4, 4], [1, 1, 3, 4], [0, 0, 0, 0]], dtype=torch.float32)
        box = bitmask.get_bounding_boxes()
        self.assertTrue(torch.all(box.tensor == box_true).item())

        for box in box_true:
            poly = box[[0, 1, 2, 1, 2, 3, 0, 3]].numpy()
            mask = polygons_to_bitmask([poly], 4, 4)
            reconstruct_box = BitMasks(mask[None, :, :]).get_bounding_boxes()[0].tensor
            self.assertTrue(torch.all(box == reconstruct_box).item())

            reconstruct_box = PolygonMasks([[poly]]).get_bounding_boxes()[0].tensor
            self.assertTrue(torch.all(box == reconstruct_box).item())

    def test_from_empty_polygons(self):
        masks = BitMasks.from_polygon_masks([], 100, 100)
        self.assertEqual(masks.tensor.shape, (0, 100, 100))


if __name__ == "__main__":
    unittest.main()
