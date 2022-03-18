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
from torch import nn

from detectron2.config import CfgNode
from detectron2.structures import Instances

from .cycle_pix2shape import PixToShapeCycleLoss
from .cycle_shape2shape import ShapeToShapeCycleLoss
from .embed import EmbeddingLoss
from .embed_utils import CseAnnotationsAccumulator
from .mask_or_segm import MaskOrSegmentationLoss
from .registry import DENSEPOSE_LOSS_REGISTRY
from .soft_embed import SoftEmbeddingLoss
from .utils import BilinearInterpolationHelper, LossDict, extract_packed_annotations_from_matches


@DENSEPOSE_LOSS_REGISTRY.register()
class DensePoseCseLoss:
    """ """

    _EMBED_LOSS_REGISTRY = {
        EmbeddingLoss.__name__: EmbeddingLoss,
        SoftEmbeddingLoss.__name__: SoftEmbeddingLoss,
    }

    def __init__(self, cfg: CfgNode):
        """
        Initialize CSE loss from configuration options

        Args:
            cfg (CfgNode): configuration options
        """
        self.w_segm = cfg.MODEL.ROI_DENSEPOSE_HEAD.INDEX_WEIGHTS
        self.w_embed = cfg.MODEL.ROI_DENSEPOSE_HEAD.CSE.EMBED_LOSS_WEIGHT
        self.segm_loss = MaskOrSegmentationLoss(cfg)
        self.embed_loss = DensePoseCseLoss.create_embed_loss(cfg)
        self.do_shape2shape = cfg.MODEL.ROI_DENSEPOSE_HEAD.CSE.SHAPE_TO_SHAPE_CYCLE_LOSS.ENABLED
        if self.do_shape2shape:
            self.w_shape2shape = cfg.MODEL.ROI_DENSEPOSE_HEAD.CSE.SHAPE_TO_SHAPE_CYCLE_LOSS.WEIGHT
            self.shape2shape_loss = ShapeToShapeCycleLoss(cfg)
        self.do_pix2shape = cfg.MODEL.ROI_DENSEPOSE_HEAD.CSE.PIX_TO_SHAPE_CYCLE_LOSS.ENABLED
        if self.do_pix2shape:
            self.w_pix2shape = cfg.MODEL.ROI_DENSEPOSE_HEAD.CSE.PIX_TO_SHAPE_CYCLE_LOSS.WEIGHT
            self.pix2shape_loss = PixToShapeCycleLoss(cfg)

    @classmethod
    def create_embed_loss(cls, cfg: CfgNode):
        # registry not used here, since embedding losses are currently local
        # and are not used anywhere else
        return cls._EMBED_LOSS_REGISTRY[cfg.MODEL.ROI_DENSEPOSE_HEAD.CSE.EMBED_LOSS_NAME](cfg)

    def __call__(
        self,
        proposals_with_gt: List[Instances],
        densepose_predictor_outputs: Any,
        embedder: nn.Module,
    ) -> LossDict:
        if not len(proposals_with_gt):
            return self.produce_fake_losses(densepose_predictor_outputs, embedder)
        accumulator = CseAnnotationsAccumulator()
        packed_annotations = extract_packed_annotations_from_matches(proposals_with_gt, accumulator)
        if packed_annotations is None:
            return self.produce_fake_losses(densepose_predictor_outputs, embedder)
        h, w = densepose_predictor_outputs.embedding.shape[2:]
        interpolator = BilinearInterpolationHelper.from_matches(
            packed_annotations,
            (h, w),
        )
        meshid_to_embed_losses = self.embed_loss(
            proposals_with_gt,
            densepose_predictor_outputs,
            packed_annotations,
            interpolator,
            embedder,
        )
        embed_loss_dict = {
            f"loss_densepose_E{meshid}": self.w_embed * meshid_to_embed_losses[meshid]
            for meshid in meshid_to_embed_losses
        }
        all_loss_dict = {
            "loss_densepose_S": self.w_segm
            * self.segm_loss(proposals_with_gt, densepose_predictor_outputs, packed_annotations),
            **embed_loss_dict,
        }
        if self.do_shape2shape:
            all_loss_dict["loss_shape2shape"] = self.w_shape2shape * self.shape2shape_loss(embedder)
        if self.do_pix2shape:
            all_loss_dict["loss_pix2shape"] = self.w_pix2shape * self.pix2shape_loss(
                proposals_with_gt, densepose_predictor_outputs, packed_annotations, embedder
            )
        return all_loss_dict

    def produce_fake_losses(
        self, densepose_predictor_outputs: Any, embedder: nn.Module
    ) -> LossDict:
        meshname_to_embed_losses = self.embed_loss.fake_values(
            densepose_predictor_outputs, embedder=embedder
        )
        embed_loss_dict = {
            f"loss_densepose_E{mesh_name}": meshname_to_embed_losses[mesh_name]
            for mesh_name in meshname_to_embed_losses
        }
        all_loss_dict = {
            "loss_densepose_S": self.segm_loss.fake_value(densepose_predictor_outputs),
            **embed_loss_dict,
        }
        if self.do_shape2shape:
            all_loss_dict["loss_shape2shape"] = self.shape2shape_loss.fake_value(embedder)
        if self.do_pix2shape:
            all_loss_dict["loss_pix2shape"] = self.pix2shape_loss.fake_value(
                densepose_predictor_outputs, embedder
            )
        return all_loss_dict
