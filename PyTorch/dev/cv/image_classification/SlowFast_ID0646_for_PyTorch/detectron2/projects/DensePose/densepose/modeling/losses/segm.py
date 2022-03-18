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
from torch.nn import functional as F

from detectron2.config import CfgNode
from detectron2.structures import Instances

from .utils import resample_data


class SegmentationLoss:
    """
    Segmentation loss as cross-entropy for raw unnormalized scores given ground truth
    labels. Segmentation ground truth labels are defined for the bounding box of
    interest at some fixed resolution [S, S], where
        S = MODEL.ROI_DENSEPOSE_HEAD.HEATMAP_SIZE.
    """

    def __init__(self, cfg: CfgNode):
        """
        Initialize segmentation loss from configuration options

        Args:
            cfg (CfgNode): configuration options
        """
        self.heatmap_size = cfg.MODEL.ROI_DENSEPOSE_HEAD.HEATMAP_SIZE
        self.n_segm_chan = cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS

    def __call__(
        self,
        proposals_with_gt: List[Instances],
        densepose_predictor_outputs: Any,
        packed_annotations: Any,
    ) -> torch.Tensor:
        """
        Compute segmentation loss as cross-entropy on aligned segmentation
        ground truth and estimated scores.

        Args:
            proposals_with_gt (list of Instances): detections with associated ground truth data
            densepose_predictor_outputs: an object of a dataclass that contains predictor outputs
                with estimated values; assumed to have the following attributes:
                * coarse_segm - coarse segmentation estimates, tensor of shape [N, D, S, S]
            packed_annotations: packed annotations for efficient loss computation;
                the following attributes are used:
                 - coarse_segm_gt
                 - bbox_xywh_gt
                 - bbox_xywh_est
        """
        if packed_annotations.coarse_segm_gt is None:
            return self.fake_value(densepose_predictor_outputs)
        coarse_segm_est = densepose_predictor_outputs.coarse_segm[packed_annotations.bbox_indices]
        with torch.no_grad():
            coarse_segm_gt = resample_data(
                packed_annotations.coarse_segm_gt.unsqueeze(1),
                packed_annotations.bbox_xywh_gt,
                packed_annotations.bbox_xywh_est,
                self.heatmap_size,
                self.heatmap_size,
                mode="nearest",
                padding_mode="zeros",
            ).squeeze(1)
        if self.n_segm_chan == 2:
            coarse_segm_gt = coarse_segm_gt > 0
        return F.cross_entropy(coarse_segm_est, coarse_segm_gt.long())

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
