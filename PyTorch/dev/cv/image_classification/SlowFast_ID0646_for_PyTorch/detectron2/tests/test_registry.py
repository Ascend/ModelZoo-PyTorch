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
