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

from detectron2.modeling.meta_arch import GeneralizedRCNN
from detectron2.utils.registry import _convert_target_to_string, locate


class A:
    class B:
        pass


class TestLocate(unittest.TestCase):
    def _test_obj(self, obj):
        name = _convert_target_to_string(obj)
        newobj = locate(name)
        self.assertIs(obj, newobj)

    def test_basic(self):
        self._test_obj(GeneralizedRCNN)

    def test_inside_class(self):
        # requires using __qualname__ instead of __name__
        self._test_obj(A.B)

    def test_builtin(self):
        self._test_obj(len)
        self._test_obj(dict)

    def test_pytorch_optim(self):
        # pydoc.locate does not work for it
        self._test_obj(torch.optim.SGD)

    def test_failure(self):
        with self.assertRaises(ImportError):
            locate("asdf")

    def test_compress_target(self):
        from detectron2.data.transforms import RandomCrop

        name = _convert_target_to_string(RandomCrop)
        # name shouldn't contain 'augmentation_impl'
        self.assertEqual(name, "detectron2.data.transforms.RandomCrop")
        self.assertIs(RandomCrop, locate(name))
