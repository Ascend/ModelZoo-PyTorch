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

from dataclasses import dataclass
from enum import Enum

from detectron2.config import CfgNode


class DensePoseUVConfidenceType(Enum):
    """
    Statistical model type for confidence learning, possible values:
     - "iid_iso": statistically independent identically distributed residuals
         with anisotropic covariance
     - "indep_aniso": statistically independent residuals with anisotropic
         covariances
    For details, see:
    N. Neverova, D. Novotny, A. Vedaldi "Correlated Uncertainty for Learning
    Dense Correspondences from Noisy Labels", p. 918--926, in Proc. NIPS 2019
    """

    # fmt: off
    IID_ISO     = "iid_iso"
    INDEP_ANISO = "indep_aniso"
    # fmt: on


@dataclass
class DensePoseUVConfidenceConfig:
    """
    Configuration options for confidence on UV data
    """

    enabled: bool = False
    # lower bound on UV confidences
    epsilon: float = 0.01
    type: DensePoseUVConfidenceType = DensePoseUVConfidenceType.IID_ISO


@dataclass
class DensePoseSegmConfidenceConfig:
    """
    Configuration options for confidence on segmentation
    """

    enabled: bool = False
    # lower bound on confidence values
    epsilon: float = 0.01


@dataclass
class DensePoseConfidenceModelConfig:
    """
    Configuration options for confidence models
    """

    # confidence for U and V values
    uv_confidence: DensePoseUVConfidenceConfig
    # segmentation confidence
    segm_confidence: DensePoseSegmConfidenceConfig

    @staticmethod
    def from_cfg(cfg: CfgNode) -> "DensePoseConfidenceModelConfig":
        return DensePoseConfidenceModelConfig(
            uv_confidence=DensePoseUVConfidenceConfig(
                enabled=cfg.MODEL.ROI_DENSEPOSE_HEAD.UV_CONFIDENCE.ENABLED,
                epsilon=cfg.MODEL.ROI_DENSEPOSE_HEAD.UV_CONFIDENCE.EPSILON,
                type=DensePoseUVConfidenceType(cfg.MODEL.ROI_DENSEPOSE_HEAD.UV_CONFIDENCE.TYPE),
            ),
            segm_confidence=DensePoseSegmConfidenceConfig(
                enabled=cfg.MODEL.ROI_DENSEPOSE_HEAD.SEGM_CONFIDENCE.ENABLED,
                epsilon=cfg.MODEL.ROI_DENSEPOSE_HEAD.SEGM_CONFIDENCE.EPSILON,
            ),
        )
