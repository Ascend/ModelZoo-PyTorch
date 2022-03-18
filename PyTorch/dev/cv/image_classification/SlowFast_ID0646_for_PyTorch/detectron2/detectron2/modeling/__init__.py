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
from detectron2.layers import ShapeSpec

from .anchor_generator import build_anchor_generator, ANCHOR_GENERATOR_REGISTRY
from .backbone import (
    BACKBONE_REGISTRY,
    FPN,
    Backbone,
    ResNet,
    ResNetBlockBase,
    build_backbone,
    build_resnet_backbone,
    make_stage,
)
from .meta_arch import (
    META_ARCH_REGISTRY,
    SEM_SEG_HEADS_REGISTRY,
    GeneralizedRCNN,
    PanopticFPN,
    ProposalNetwork,
    RetinaNet,
    SemanticSegmentor,
    build_model,
    build_sem_seg_head,
)
from .postprocessing import detector_postprocess
from .proposal_generator import (
    PROPOSAL_GENERATOR_REGISTRY,
    build_proposal_generator,
    RPN_HEAD_REGISTRY,
    build_rpn_head,
)
from .roi_heads import (
    ROI_BOX_HEAD_REGISTRY,
    ROI_HEADS_REGISTRY,
    ROI_KEYPOINT_HEAD_REGISTRY,
    ROI_MASK_HEAD_REGISTRY,
    ROIHeads,
    StandardROIHeads,
    BaseMaskRCNNHead,
    BaseKeypointRCNNHead,
    FastRCNNOutputLayers,
    build_box_head,
    build_keypoint_head,
    build_mask_head,
    build_roi_heads,
)
from .test_time_augmentation import DatasetMapperTTA, GeneralizedRCNNWithTTA
from .mmdet_wrapper import MMDetBackbone, MMDetDetector

_EXCLUDE = {"ShapeSpec"}
__all__ = [k for k in globals().keys() if k not in _EXCLUDE and not k.startswith("_")]


from detectron2.utils.env import fixup_module_metadata

fixup_module_metadata(__name__, globals(), __all__)
del fixup_module_metadata
