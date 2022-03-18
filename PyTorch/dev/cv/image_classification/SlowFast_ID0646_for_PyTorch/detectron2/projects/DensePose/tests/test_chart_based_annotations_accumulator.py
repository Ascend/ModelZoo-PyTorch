#
# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================
#
# Copyright (c) Facebook, Inc. and its affiliates.

import unittest
import torch

from detectron2.structures import Boxes, BoxMode, Instances

from densepose.modeling.losses.utils import ChartBasedAnnotationsAccumulator
from densepose.structures import DensePoseDataRelative, DensePoseList

image_shape = (100, 100)
instances = Instances(image_shape)
n_instances = 3
instances.proposal_boxes = Boxes(torch.rand(n_instances, 4))
instances.gt_boxes = Boxes(torch.rand(n_instances, 4))


# instances.gt_densepose = None cannot happen because instances attributes need a length
class TestChartBasedAnnotationsAccumulator(unittest.TestCase):
    def test_chart_based_annotations_accumulator_no_gt_densepose(self):
        accumulator = ChartBasedAnnotationsAccumulator()
        accumulator.accumulate(instances)
        expected_values = {"nxt_bbox_with_dp_index": 0, "nxt_bbox_index": n_instances}
        for key in accumulator.__dict__:
            self.assertEqual(getattr(accumulator, key), expected_values.get(key, []))

    def test_chart_based_annotations_accumulator_gt_densepose_none(self):
        instances.gt_densepose = [None] * n_instances
        accumulator = ChartBasedAnnotationsAccumulator()
        accumulator.accumulate(instances)
        expected_values = {"nxt_bbox_with_dp_index": 0, "nxt_bbox_index": n_instances}
        for key in accumulator.__dict__:
            self.assertEqual(getattr(accumulator, key), expected_values.get(key, []))

    def test_chart_based_annotations_accumulator_gt_densepose(self):
        data_relative_keys = [
            DensePoseDataRelative.X_KEY,
            DensePoseDataRelative.Y_KEY,
            DensePoseDataRelative.I_KEY,
            DensePoseDataRelative.U_KEY,
            DensePoseDataRelative.V_KEY,
            DensePoseDataRelative.S_KEY,
        ]
        annotations = [DensePoseDataRelative({k: [0] for k in data_relative_keys})] * n_instances
        instances.gt_densepose = DensePoseList(annotations, instances.gt_boxes, image_shape)
        accumulator = ChartBasedAnnotationsAccumulator()
        accumulator.accumulate(instances)
        bbox_xywh_est = BoxMode.convert(
            instances.proposal_boxes.tensor.clone(), BoxMode.XYXY_ABS, BoxMode.XYWH_ABS
        )
        bbox_xywh_gt = BoxMode.convert(
            instances.gt_boxes.tensor.clone(), BoxMode.XYXY_ABS, BoxMode.XYWH_ABS
        )
        expected_values = {
            "s_gt": [
                torch.zeros((3, DensePoseDataRelative.MASK_SIZE, DensePoseDataRelative.MASK_SIZE))
            ]
            * n_instances,
            "bbox_xywh_est": bbox_xywh_est.split(1),
            "bbox_xywh_gt": bbox_xywh_gt.split(1),
            "point_bbox_with_dp_indices": [torch.tensor([i]) for i in range(n_instances)],
            "point_bbox_indices": [torch.tensor([i]) for i in range(n_instances)],
            "bbox_indices": list(range(n_instances)),
            "nxt_bbox_with_dp_index": n_instances,
            "nxt_bbox_index": n_instances,
        }
        default_value = [torch.tensor([0])] * 3
        for key in accumulator.__dict__:
            to_test = getattr(accumulator, key)
            gt_value = expected_values.get(key, default_value)
            if key in ["nxt_bbox_with_dp_index", "nxt_bbox_index"]:
                self.assertEqual(to_test, gt_value)
            elif key == "bbox_indices":
                self.assertListEqual(to_test, gt_value)
            else:
                self.assertTrue(torch.allclose(torch.stack(to_test), torch.stack(gt_value)))
