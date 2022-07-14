# MIT License
#
# Copyright (c) 2020 xxx
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ============================================================================
#
# Copyright 2021 Huawei Technologies Co., Ltd
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random
from random import randint
from random import shuffle
import torchvision

class SwapReferenceTest(object):
    def __call__(self, sample):
        prob = random.random()
        # Half chance to swap reference and test
        if prob > 0.5:
            trf_reference = sample['reference']
            trf_test = sample['test']
        else:
            trf_reference = sample['test']
            trf_test = sample['reference']
                
        return trf_reference, trf_test

class JitterGamma(object):
    def __call__(self, sample):
        prob = random.random()
        trf_reference = sample['reference']
        trf_test = sample['test']
        # Half chance to swap reference and test
        if prob > 0.5:
            gamma = random.random() + 0.1
            trf_reference = torchvision.transforms.functional.adjust_gamma(trf_reference, gamma)
            trf_test = torchvision.transforms.functional.adjust_gamma(trf_test, gamma)
                
        return trf_reference, trf_test