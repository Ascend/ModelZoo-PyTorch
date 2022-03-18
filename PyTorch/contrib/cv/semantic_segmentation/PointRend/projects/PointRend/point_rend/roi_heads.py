# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
