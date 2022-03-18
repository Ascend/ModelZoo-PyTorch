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
from __future__ import absolute_import, division, print_function, unicode_literals
import unittest
import torch

from detectron2.layers import batched_nms
from detectron2.utils.testing import random_boxes


class TestNMS(unittest.TestCase):
    def _create_tensors(self, N):
        boxes = random_boxes(N, 200)
        scores = torch.rand(N)
        return boxes, scores

    def test_nms_scriptability(self):
        N = 2000
        num_classes = 50
        boxes, scores = self._create_tensors(N)
        idxs = torch.randint(0, num_classes, (N,))
        scripted_batched_nms = torch.jit.script(batched_nms)
        err_msg = "NMS is incompatible with jit-scripted NMS for IoU={}"

        for iou in [0.2, 0.5, 0.8]:
            keep_ref = batched_nms(boxes, scores, idxs, iou)
            backup = boxes.clone()
            scripted_keep = scripted_batched_nms(boxes, scores, idxs, iou)
            assert torch.allclose(boxes, backup), "boxes modified by jit-scripted batched_nms"
            self.assertTrue(torch.equal(keep_ref, scripted_keep), err_msg.format(iou))


if __name__ == "__main__":
    unittest.main()
