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

import random
from collections.abc import Callable
from enum import Enum
from typing import Callable as TCallable
from typing import List

FrameTsList = List[int]
FrameSelector = TCallable[[FrameTsList], FrameTsList]


class FrameSelectionStrategy(Enum):
    """
    Frame selection strategy used with videos:
     - "random_k": select k random frames
     - "first_k": select k first frames
     - "last_k": select k last frames
     - "all": select all frames
    """

    # fmt: off
    RANDOM_K = "random_k"
    FIRST_K  = "first_k"
    LAST_K   = "last_k"
    ALL      = "all"
    # fmt: on


class RandomKFramesSelector(Callable):  # pyre-ignore[39]
    """
    Selector that retains at most `k` random frames
    """

    def __init__(self, k: int):
        self.k = k

    def __call__(self, frame_tss: FrameTsList) -> FrameTsList:
        """
        Select `k` random frames

        Args:
          frames_tss (List[int]): timestamps of input frames
        Returns:
          List[int]: timestamps of selected frames
        """
        return random.sample(frame_tss, min(self.k, len(frame_tss)))


class FirstKFramesSelector(Callable):  # pyre-ignore[39]
    """
    Selector that retains at most `k` first frames
    """

    def __init__(self, k: int):
        self.k = k

    def __call__(self, frame_tss: FrameTsList) -> FrameTsList:
        """
        Select `k` first frames

        Args:
          frames_tss (List[int]): timestamps of input frames
        Returns:
          List[int]: timestamps of selected frames
        """
        return frame_tss[: self.k]


class LastKFramesSelector(Callable):  # pyre-ignore[39]
    """
    Selector that retains at most `k` last frames from video data
    """

    def __init__(self, k: int):
        self.k = k

    def __call__(self, frame_tss: FrameTsList) -> FrameTsList:
        """
        Select `k` last frames

        Args:
          frames_tss (List[int]): timestamps of input frames
        Returns:
          List[int]: timestamps of selected frames
        """
        return frame_tss[-self.k :]
