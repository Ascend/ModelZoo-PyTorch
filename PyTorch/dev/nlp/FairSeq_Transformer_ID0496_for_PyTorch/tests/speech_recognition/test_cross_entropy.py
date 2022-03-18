#!/usr/bin/env python3
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
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from examples.speech_recognition.criterions.cross_entropy_acc import CrossEntropyWithAccCriterion
from .asr_test_base import CrossEntropyCriterionTestBase


class CrossEntropyWithAccCriterionTest(CrossEntropyCriterionTestBase):
    def setUp(self):
        self.criterion_cls = CrossEntropyWithAccCriterion
        super().setUp()

    def test_cross_entropy_all_correct(self):
        sample = self.get_test_sample(correct=True, soft_target=False, aggregate=False)
        loss, sample_size, logging_output = self.criterion(
            self.model, sample, "sum", log_probs=True
        )
        assert logging_output["correct"] == 20
        assert logging_output["total"] == 20
        assert logging_output["sample_size"] == 20
        assert logging_output["ntokens"] == 20

    def test_cross_entropy_all_wrong(self):
        sample = self.get_test_sample(correct=False, soft_target=False, aggregate=False)
        loss, sample_size, logging_output = self.criterion(
            self.model, sample, "sum", log_probs=True
        )
        assert logging_output["correct"] == 0
        assert logging_output["total"] == 20
        assert logging_output["sample_size"] == 20
        assert logging_output["ntokens"] == 20
