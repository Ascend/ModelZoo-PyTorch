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
import logging

from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads


@ROI_HEADS_REGISTRY.register()
class PointRendROIHeads(StandardROIHeads):
    """
    Identical to StandardROIHeads, except for some weights conversion code to
    handle old models.
    """

    _version = 2

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            logger = logging.getLogger(__name__)
            logger.warning(
                "Weight format of PointRend models have changed! "
                "Please upgrade your models. Applying automatic conversion now ..."
            )
            for k in list(state_dict.keys()):
                newk = k
                if k.startswith(prefix + "mask_point_head"):
                    newk = k.replace(prefix + "mask_point_head", prefix + "mask_head.point_head")
                if k.startswith(prefix + "mask_coarse_head"):
                    newk = k.replace(prefix + "mask_coarse_head", prefix + "mask_head.coarse_head")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]

    @classmethod
    def _init_mask_head(cls, cfg, input_shape):
        if cfg.MODEL.MASK_ON and cfg.MODEL.ROI_MASK_HEAD.NAME != "PointRendMaskHead":
            logger = logging.getLogger(__name__)
            logger.warning(
                "Config of PointRend models have changed! "
                "Please upgrade your models. Applying automatic conversion now ..."
            )
            assert cfg.MODEL.ROI_MASK_HEAD.NAME == "CoarseMaskHead"
            cfg.defrost()
            cfg.MODEL.ROI_MASK_HEAD.NAME = "PointRendMaskHead"
            cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE = ""
            cfg.freeze()
        return super()._init_mask_head(cfg, input_shape)
