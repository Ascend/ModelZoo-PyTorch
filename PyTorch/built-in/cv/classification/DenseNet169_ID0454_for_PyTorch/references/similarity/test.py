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
#import unittest
from collections import defaultdict

from torch.utils.data import DataLoader
from torchvision.datasets import FakeData
import torchvision.transforms as transforms

from sampler import PKSampler


class Tester(unittest.TestCase):

    def test_pksampler(self):
        p, k = 16, 4

        # Ensure sampler does not allow p to be greater than num_classes
        dataset = FakeData(size=100, num_classes=10, image_size=(3, 1, 1))
        targets = [target.item() for _, target in dataset]
        self.assertRaises(AssertionError, PKSampler, targets, p, k)

        # Ensure p, k constraints on batch
        dataset = FakeData(size=1000, num_classes=100, image_size=(3, 1, 1),
                           transform=transforms.ToTensor())
        targets = [target.item() for _, target in dataset]
        sampler = PKSampler(targets, p, k)
        loader = DataLoader(dataset, batch_size=p * k, sampler=sampler)

        for _, labels in loader:
            bins = defaultdict(int)
            for l in labels.tolist():
                bins[l] += 1

            # Ensure that each batch has samples from exactly p classes
            self.assertEqual(len(bins), p)

            # Ensure that there are k samples from each class
            for l in bins:
                self.assertEqual(bins[l], k)


if __name__ == '__main__':
    unittest.main()
