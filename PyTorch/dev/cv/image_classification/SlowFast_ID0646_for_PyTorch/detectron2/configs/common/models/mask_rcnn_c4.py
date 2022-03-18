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
from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.modeling.meta_arch import GeneralizedRCNN
from detectron2.modeling.anchor_generator import DefaultAnchorGenerator
from detectron2.modeling.backbone import BasicStem, BottleneckBlock, ResNet
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.proposal_generator import RPN, StandardRPNHead
from detectron2.modeling.roi_heads import (
    FastRCNNOutputLayers,
    MaskRCNNConvUpsampleHead,
    Res5ROIHeads,
)

model = L(GeneralizedRCNN)(
    backbone=L(ResNet)(
        stem=L(BasicStem)(in_channels=3, out_channels=64, norm="FrozenBN"),
        stages=L(ResNet.make_default_stages)(
            depth=50,
            stride_in_1x1=True,
            norm="FrozenBN",
        ),
        out_features=["res4"],
    ),
    proposal_generator=L(RPN)(
        in_features=["res4"],
        head=L(StandardRPNHead)(in_channels=1024, num_anchors=15),
        anchor_generator=L(DefaultAnchorGenerator)(
            sizes=[[32, 64, 128, 256, 512]],
            aspect_ratios=[0.5, 1.0, 2.0],
            strides=[16],
            offset=0.0,
        ),
        anchor_matcher=L(Matcher)(
            thresholds=[0.3, 0.7], labels=[0, -1, 1], allow_low_quality_matches=True
        ),
        box2box_transform=L(Box2BoxTransform)(weights=[1.0, 1.0, 1.0, 1.0]),
        batch_size_per_image=256,
        positive_fraction=0.5,
        pre_nms_topk=(12000, 6000),
        post_nms_topk=(2000, 1000),
        nms_thresh=0.7,
    ),
    roi_heads=L(Res5ROIHeads)(
        num_classes=80,
        batch_size_per_image=512,
        positive_fraction=0.25,
        proposal_matcher=L(Matcher)(
            thresholds=[0.5], labels=[0, 1], allow_low_quality_matches=False
        ),
        in_features=["res4"],
        pooler=L(ROIPooler)(
            output_size=14,
            scales=(1.0 / 16,),
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
        ),
        res5=L(ResNet.make_stage)(
            block_class=BottleneckBlock,
            num_blocks=3,
            stride_per_block=[2, 1, 1],
            in_channels=1024,
            bottleneck_channels=512,
            out_channels=2048,
            norm="FrozenBN",
            stride_in_1x1=True,
        ),
        box_predictor=L(FastRCNNOutputLayers)(
            input_shape=L(ShapeSpec)(channels="${...res5.out_channels}", height=1, width=1),
            test_score_thresh=0.05,
            box2box_transform=L(Box2BoxTransform)(weights=(10, 10, 5, 5)),
            num_classes="${..num_classes}",
        ),
        mask_head=L(MaskRCNNConvUpsampleHead)(
            input_shape=L(ShapeSpec)(
                channels="${...res5.out_channels}",
                width="${...pooler.output_size}",
                height="${...pooler.output_size}",
            ),
            num_classes="${..num_classes}",
            conv_dims=[256],
        ),
    ),
    pixel_mean=[103.530, 116.280, 123.675],
    pixel_std=[1.0, 1.0, 1.0],
    input_format="BGR",
)
