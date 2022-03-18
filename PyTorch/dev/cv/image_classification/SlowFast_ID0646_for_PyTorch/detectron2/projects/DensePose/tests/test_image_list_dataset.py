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

import contextlib
import os
import tempfile
import unittest
import torch
from torchvision.utils import save_image

from densepose.data.image_list_dataset import ImageListDataset
from densepose.data.transform import ImageResizeTransform


@contextlib.contextmanager
def temp_image(height, width):
    random_image = torch.rand(height, width)
    with tempfile.NamedTemporaryFile(suffix=".jpg") as f:
        f.close()
        save_image(random_image, f.name)
        yield f.name
    os.unlink(f.name)


class TestImageListDataset(unittest.TestCase):
    def test_image_list_dataset(self):
        height, width = 720, 1280
        with temp_image(height, width) as image_fpath:
            image_list = [image_fpath]
            category_list = [None]
            dataset = ImageListDataset(image_list, category_list)
            self.assertEqual(len(dataset), 1)
            data1, categories1 = dataset[0]["images"], dataset[0]["categories"]
            self.assertEqual(data1.shape, torch.Size((1, 3, height, width)))
            self.assertEqual(data1.dtype, torch.float32)
            self.assertIsNone(categories1[0])

    def test_image_list_dataset_with_transform(self):
        height, width = 720, 1280
        with temp_image(height, width) as image_fpath:
            image_list = [image_fpath]
            category_list = [None]
            transform = ImageResizeTransform()
            dataset = ImageListDataset(image_list, category_list, transform)
            self.assertEqual(len(dataset), 1)
            data1, categories1 = dataset[0]["images"], dataset[0]["categories"]
            self.assertEqual(data1.shape, torch.Size((1, 3, 749, 1333)))
            self.assertEqual(data1.dtype, torch.float32)
            self.assertIsNone(categories1[0])
