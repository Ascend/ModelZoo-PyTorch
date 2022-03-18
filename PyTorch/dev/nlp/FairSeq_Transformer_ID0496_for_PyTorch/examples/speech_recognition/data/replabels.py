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

"""
Replabel transforms for use with wav2letter's ASG criterion.
"""


def replabel_symbol(i):
    """
    Replabel symbols used in wav2letter, currently just "1", "2", ...
    This prevents training with numeral tokens, so this might change in the future
    """
    return str(i)


def pack_replabels(tokens, dictionary, max_reps):
    """
    Pack a token sequence so that repeated symbols are replaced by replabels
    """
    if len(tokens) == 0 or max_reps <= 0:
        return tokens

    replabel_value_to_idx = [0] * (max_reps + 1)
    for i in range(1, max_reps + 1):
        replabel_value_to_idx[i] = dictionary.index(replabel_symbol(i))

    result = []
    prev_token = -1
    num_reps = 0
    for token in tokens:
        if token == prev_token and num_reps < max_reps:
            num_reps += 1
        else:
            if num_reps > 0:
                result.append(replabel_value_to_idx[num_reps])
                num_reps = 0
            result.append(token)
            prev_token = token
    if num_reps > 0:
        result.append(replabel_value_to_idx[num_reps])
    return result


def unpack_replabels(tokens, dictionary, max_reps):
    """
    Unpack a token sequence so that replabels are replaced by repeated symbols
    """
    if len(tokens) == 0 or max_reps <= 0:
        return tokens

    replabel_idx_to_value = {}
    for i in range(1, max_reps + 1):
        replabel_idx_to_value[dictionary.index(replabel_symbol(i))] = i

    result = []
    prev_token = -1
    for token in tokens:
        try:
            for _ in range(replabel_idx_to_value[token]):
                result.append(prev_token)
            prev_token = -1
        except KeyError:
            result.append(token)
            prev_token = token
    return result
