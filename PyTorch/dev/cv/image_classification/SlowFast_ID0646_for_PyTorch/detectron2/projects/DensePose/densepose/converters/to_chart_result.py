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

from typing import Any

from detectron2.structures import Boxes

from ..structures import DensePoseChartResult, DensePoseChartResultWithConfidences
from .base import BaseConverter


class ToChartResultConverter(BaseConverter):
    """
    Converts various DensePose predictor outputs to DensePose results.
    Each DensePose predictor output type has to register its convertion strategy.
    """

    registry = {}
    dst_type = DensePoseChartResult

    @classmethod
    def convert(cls, predictor_outputs: Any, boxes: Boxes, *args, **kwargs) -> DensePoseChartResult:
        """
        Convert DensePose predictor outputs to DensePoseResult using some registered
        converter. Does recursive lookup for base classes, so there's no need
        for explicit registration for derived classes.

        Args:
            densepose_predictor_outputs: DensePose predictor output to be
                converted to BitMasks
            boxes (Boxes): bounding boxes that correspond to the DensePose
                predictor outputs
        Return:
            An instance of DensePoseResult. If no suitable converter was found, raises KeyError
        """
        return super(ToChartResultConverter, cls).convert(predictor_outputs, boxes, *args, **kwargs)


class ToChartResultConverterWithConfidences(BaseConverter):
    """
    Converts various DensePose predictor outputs to DensePose results.
    Each DensePose predictor output type has to register its convertion strategy.
    """

    registry = {}
    dst_type = DensePoseChartResultWithConfidences

    @classmethod
    def convert(
        cls, predictor_outputs: Any, boxes: Boxes, *args, **kwargs
    ) -> DensePoseChartResultWithConfidences:
        """
        Convert DensePose predictor outputs to DensePoseResult with confidences
        using some registered converter. Does recursive lookup for base classes,
        so there's no need for explicit registration for derived classes.

        Args:
            densepose_predictor_outputs: DensePose predictor output with confidences
                to be converted to BitMasks
            boxes (Boxes): bounding boxes that correspond to the DensePose
                predictor outputs
        Return:
            An instance of DensePoseResult. If no suitable converter was found, raises KeyError
        """
        return super(ToChartResultConverterWithConfidences, cls).convert(
            predictor_outputs, boxes, *args, **kwargs
        )
