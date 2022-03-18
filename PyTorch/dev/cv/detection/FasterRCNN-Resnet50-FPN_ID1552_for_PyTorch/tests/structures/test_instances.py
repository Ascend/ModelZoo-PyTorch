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

from detectron2.export.torchscript import patch_instances
from detectron2.structures import Instances
from detectron2.utils.env import TORCH_VERSION


class TestInstances(unittest.TestCase):
    def test_int_indexing(self):
        attr1 = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 0.5], [0.0, 0.0, 1.0], [0.0, 0.5, 0.5]])
        attr2 = torch.tensor([0.1, 0.2, 0.3, 0.4])
        instances = Instances((100, 100))
        instances.attr1 = attr1
        instances.attr2 = attr2
        for i in range(-len(instances), len(instances)):
            inst = instances[i]
            self.assertEqual((inst.attr1 == attr1[i]).all(), True)
            self.assertEqual((inst.attr2 == attr2[i]).all(), True)

        self.assertRaises(IndexError, lambda: instances[len(instances)])
        self.assertRaises(IndexError, lambda: instances[-len(instances) - 1])

    @unittest.skipIf(TORCH_VERSION < (1, 6), "Insufficient pytorch version")
    def test_script_new_fields(self):
        class f(torch.nn.Module):
            def forward(self, x: Instances):
                proposal_boxes = x.proposal_boxes  # noqa F841
                objectness_logits = x.objectness_logits  # noqa F841
                return x

        class g(torch.nn.Module):
            def forward(self, x: Instances):
                mask = x.mask  # noqa F841
                return x

        class g2(torch.nn.Module):
            def forward(self, x: Instances):
                proposal_boxes = x.proposal_boxes  # noqa F841
                return x

        fields = {"proposal_boxes": "Boxes", "objectness_logits": "Tensor"}
        with patch_instances(fields):
            torch.jit.script(f())

        # can't script anymore after exiting the context
        with self.assertRaises(Exception):
            torch.jit.script(g2())

        new_fields = {"mask": "Tensor"}
        with patch_instances(new_fields):
            torch.jit.script(g())
            with self.assertRaises(Exception):
                torch.jit.script(g2())


if __name__ == "__main__":
    unittest.main()
