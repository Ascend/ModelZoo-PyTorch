# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
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
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

from itertools import islice, chain

import numpy as np

from ..data import TwoStreamBatchSampler
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))

def test_two_stream_batch_sampler():
    import sys
    print(sys.version)
    sampler = TwoStreamBatchSampler(primary_indices=range(10),
                                    secondary_indices=range(-2, 0),
                                    batch_size=3,
                                    secondary_batch_size=1)
    batches = list(sampler)

    # All batches have length 3
    assert all(len(batch) == 3 for batch in batches)

    # All batches include two items from the primary batch
    assert all(len([i for i in batch if i >= 0]) == 2 for batch in batches)

    # All batches include one item from the secondary batch
    assert all(len([i for i in batch if i < 0]) == 1 for batch in batches)

    # All primary items are included in the epoch
    assert len(sampler.primary_indices) % sampler.secondary_batch_size == 0 # Pre-condition
    assert sorted(i for i in chain(*batches) if i >= 0) == list(range(10)) # Post-condition

    # Secondary items are iterated through before beginning again
    assert sorted(i for i in chain(*batches[:2]) if i < 0) == list(range(-2, 0))


def test_two_stream_batch_sampler_uneven():
    import sys
    print(sys.version)
    sampler = TwoStreamBatchSampler(primary_indices=range(11),
                                    secondary_indices=range(-3, 0),
                                    batch_size=5,
                                    secondary_batch_size=2)
    batches = list(sampler)

    # All batches have length 5
    assert all(len(batch) == 5 for batch in batches)

    # All batches include 3 items from the primary batch
    assert all(len([i for i in batch if i >= 0]) == 3 for batch in batches)

    # All batches include 2 items from the secondary batch
    assert all(len([i for i in batch if i < 0]) == 2 for batch in batches)

    # Almost all primary items are included in the epoch
    primary_items_met = [i for i in chain(*batches) if i >= 0]
    left_out = set(range(11)) - set(primary_items_met)
    assert len(left_out) == 11 % 3

    # Secondary items are iterated through before beginning again
    assert sorted(i for i in chain(*batches[:3]) if i < 0) == sorted(list(range(-3, 0)) * 2)
