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
from .box_head import ROI_BOX_HEAD_REGISTRY, build_box_head, FastRCNNConvFCHead
from .keypoint_head import (
    ROI_KEYPOINT_HEAD_REGISTRY,
    build_keypoint_head,
    BaseKeypointRCNNHead,
    KRCNNConvDeconvUpsampleHead,
)
from .mask_head import (
    ROI_MASK_HEAD_REGISTRY,
    build_mask_head,
    BaseMaskRCNNHead,
    MaskRCNNConvUpsampleHead,
)
from .roi_heads import (
    ROI_HEADS_REGISTRY,
    ROIHeads,
    Res5ROIHeads,
    StandardROIHeads,
    build_roi_heads,
    select_foreground_proposals,
)
from .cascade_rcnn import CascadeROIHeads
from .rotated_fast_rcnn import RROIHeads
from .fast_rcnn import FastRCNNOutputLayers

from . import cascade_rcnn  # isort:skip

__all__ = list(globals().keys())
