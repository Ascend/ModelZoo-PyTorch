# -*- coding: utf-8 -*-
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

from detectron2.config import CfgNode as CN


def add_pointrend_config(cfg):
    """
    Add config for PointRend.
    """
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Color augmentatition from SSD paper for semantic segmentation model during training.
    cfg.INPUT.COLOR_AUG_SSD = False

    # Names of the input feature maps to be used by a coarse mask head.
    cfg.MODEL.ROI_MASK_HEAD.IN_FEATURES = ("p2",)
    cfg.MODEL.ROI_MASK_HEAD.FC_DIM = 1024
    cfg.MODEL.ROI_MASK_HEAD.NUM_FC = 2
    # The side size of a coarse mask head prediction.
    cfg.MODEL.ROI_MASK_HEAD.OUTPUT_SIDE_RESOLUTION = 7
    # True if point head is used.
    cfg.MODEL.ROI_MASK_HEAD.POINT_HEAD_ON = False

    cfg.MODEL.POINT_HEAD = CN()
    cfg.MODEL.POINT_HEAD.NAME = "StandardPointHead"
    cfg.MODEL.POINT_HEAD.NUM_CLASSES = 80
    # Names of the input feature maps to be used by a mask point head.
    cfg.MODEL.POINT_HEAD.IN_FEATURES = ("p2",)
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.POINT_HEAD.TRAIN_NUM_POINTS = 14 * 14
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.POINT_HEAD.OVERSAMPLE_RATIO = 3
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.POINT_HEAD.IMPORTANCE_SAMPLE_RATIO = 0.75
    # Number of subdivision steps during inference.
    cfg.MODEL.POINT_HEAD.SUBDIVISION_STEPS = 5
    # Maximum number of points selected at each subdivision step (N).
    cfg.MODEL.POINT_HEAD.SUBDIVISION_NUM_POINTS = 28 * 28
    cfg.MODEL.POINT_HEAD.FC_DIM = 256
    cfg.MODEL.POINT_HEAD.NUM_FC = 3
    cfg.MODEL.POINT_HEAD.CLS_AGNOSTIC_MASK = False
    # If True, then coarse prediction features are used as inout for each layer in PointRend's MLP.
    cfg.MODEL.POINT_HEAD.COARSE_PRED_EACH_LAYER = True
    cfg.MODEL.POINT_HEAD.COARSE_SEM_SEG_HEAD_NAME = "SemSegFPNHead"

    """
    Add config for Implicit PointRend.
    """
    cfg.MODEL.IMPLICIT_POINTREND = CN()

    cfg.MODEL.IMPLICIT_POINTREND.IMAGE_FEATURE_ENABLED = True
    cfg.MODEL.IMPLICIT_POINTREND.POS_ENC_ENABLED = True

    cfg.MODEL.IMPLICIT_POINTREND.PARAMS_L2_REGULARIZER = 0.00001
