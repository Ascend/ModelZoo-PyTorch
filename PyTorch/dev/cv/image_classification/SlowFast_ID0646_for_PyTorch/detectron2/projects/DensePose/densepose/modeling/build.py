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

from typing import Optional
from torch import nn

from detectron2.config import CfgNode

from .cse.embedder import Embedder
from .filter import DensePoseDataFilter


def build_densepose_predictor(cfg: CfgNode, input_channels: int):
    """
    Create an instance of DensePose predictor based on configuration options.

    Args:
        cfg (CfgNode): configuration options
        input_channels (int): input tensor size along the channel dimension
    Return:
        An instance of DensePose predictor
    """
    from .predictors import DENSEPOSE_PREDICTOR_REGISTRY

    predictor_name = cfg.MODEL.ROI_DENSEPOSE_HEAD.PREDICTOR_NAME
    return DENSEPOSE_PREDICTOR_REGISTRY.get(predictor_name)(cfg, input_channels)


def build_densepose_data_filter(cfg: CfgNode):
    """
    Build DensePose data filter which selects data for training

    Args:
        cfg (CfgNode): configuration options

    Return:
        Callable: list(Tensor), list(Instances) -> list(Tensor), list(Instances)
        An instance of DensePose filter, which takes feature tensors and proposals
        as an input and returns filtered features and proposals
    """
    dp_filter = DensePoseDataFilter(cfg)
    return dp_filter


def build_densepose_head(cfg: CfgNode, input_channels: int):
    """
    Build DensePose head based on configurations options

    Args:
        cfg (CfgNode): configuration options
        input_channels (int): input tensor size along the channel dimension
    Return:
        An instance of DensePose head
    """
    from .roi_heads.registry import ROI_DENSEPOSE_HEAD_REGISTRY

    head_name = cfg.MODEL.ROI_DENSEPOSE_HEAD.NAME
    return ROI_DENSEPOSE_HEAD_REGISTRY.get(head_name)(cfg, input_channels)


def build_densepose_losses(cfg: CfgNode):
    """
    Build DensePose loss based on configurations options

    Args:
        cfg (CfgNode): configuration options
    Return:
        An instance of DensePose loss
    """
    from .losses import DENSEPOSE_LOSS_REGISTRY

    loss_name = cfg.MODEL.ROI_DENSEPOSE_HEAD.LOSS_NAME
    return DENSEPOSE_LOSS_REGISTRY.get(loss_name)(cfg)


def build_densepose_embedder(cfg: CfgNode) -> Optional[nn.Module]:
    """
    Build embedder used to embed mesh vertices into an embedding space.
    Embedder contains sub-embedders, one for each mesh ID.

    Args:
        cfg (cfgNode): configuration options
    Return:
        Embedding module
    """
    if cfg.MODEL.ROI_DENSEPOSE_HEAD.CSE.EMBEDDERS:
        return Embedder(cfg)
    return None
