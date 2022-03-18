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
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Any, List
import torch

from detectron2.config import CfgNode
from detectron2.structures import Instances

from .mask import MaskLoss
from .segm import SegmentationLoss


class MaskOrSegmentationLoss:
    """
    Mask or segmentation loss as cross-entropy for raw unnormalized scores
    given ground truth labels. Ground truth labels are either defined by coarse
    segmentation annotation, or by mask annotation, depending on the config
    value MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS
    """

    def __init__(self, cfg: CfgNode):
        """
        Initialize segmentation loss from configuration options

        Args:
            cfg (CfgNode): configuration options
        """
        self.segm_trained_by_masks = cfg.MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS
        if self.segm_trained_by_masks:
            self.mask_loss = MaskLoss()
        self.segm_loss = SegmentationLoss(cfg)

    def __call__(
        self,
        proposals_with_gt: List[Instances],
        densepose_predictor_outputs: Any,
        packed_annotations: Any,
    ) -> torch.Tensor:
        """
        Compute segmentation loss as cross-entropy between aligned unnormalized
        score estimates and ground truth; with ground truth given
        either by masks, or by coarse segmentation annotations.

        Args:
            proposals_with_gt (list of Instances): detections with associated ground truth data
            densepose_predictor_outputs: an object of a dataclass that contains predictor outputs
                with estimated values; assumed to have the following attributes:
                * coarse_segm - coarse segmentation estimates, tensor of shape [N, D, S, S]
            packed_annotations: packed annotations for efficient loss computation
        Return:
            tensor: loss value as cross-entropy for raw unnormalized scores
                given ground truth labels
        """
        if self.segm_trained_by_masks:
            return self.mask_loss(proposals_with_gt, densepose_predictor_outputs)
        return self.segm_loss(proposals_with_gt, densepose_predictor_outputs, packed_annotations)

    def fake_value(self, densepose_predictor_outputs: Any) -> torch.Tensor:
        """
        Fake segmentation loss used when no suitable ground truth data
        was found in a batch. The loss has a value 0 and is primarily used to
        construct the computation graph, so that `DistributedDataParallel`
        has similar graphs on all GPUs and can perform reduction properly.

        Args:
            densepose_predictor_outputs: DensePose predictor outputs, an object
                of a dataclass that is assumed to have `coarse_segm`
                attribute
        Return:
            Zero value loss with proper computation graph
        """
        return densepose_predictor_outputs.coarse_segm.sum() * 0
