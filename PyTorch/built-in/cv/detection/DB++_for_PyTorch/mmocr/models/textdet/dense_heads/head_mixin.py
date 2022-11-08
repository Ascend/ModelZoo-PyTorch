# -*- coding: utf-8 -*-
# BSD 3-Clause License
#
# Copyright (c) 2017
# All rights reserved.
# Copyright 2022 Huawei Technologies Co., Ltd
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
# ==========================================================================

# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmocr.models.builder import HEADS, build_loss, build_postprocessor
from mmocr.utils import check_argument


@HEADS.register_module()
class HeadMixin:
    """Base head class for text detection, including loss calcalation and
    postprocess.

    Args:
        loss (dict): Config to build loss.
        postprocessor (dict): Config to build postprocessor.
    """

    def __init__(self, loss, postprocessor):
        assert isinstance(loss, dict)
        assert isinstance(postprocessor, dict)

        self.loss_module = build_loss(loss)
        self.postprocessor = build_postprocessor(postprocessor)

    def resize_boundary(self, boundaries, scale_factor):
        """Rescale boundaries via scale_factor.

        Args:
            boundaries (list[list[float]]): The boundary list. Each boundary
                has :math:`2k+1` elements with :math:`k>=4`.
            scale_factor (ndarray): The scale factor of size :math:`(4,)`.

        Returns:
            list[list[float]]: The scaled boundaries.
        """
        assert check_argument.is_2dlist(boundaries)
        assert isinstance(scale_factor, np.ndarray)
        assert scale_factor.shape[0] == 4

        for b in boundaries:
            sz = len(b)
            check_argument.valid_boundary(b, True)
            b[:sz -
              1] = (np.array(b[:sz - 1]) *
                    (np.tile(scale_factor[:2], int(
                        (sz - 1) / 2)).reshape(1, sz - 1))).flatten().tolist()
        return boundaries

    def get_boundary(self, score_maps, img_metas, rescale):
        """Compute text boundaries via post processing.

        Args:
            score_maps (Tensor): The text score map.
            img_metas (dict): The image meta info.
            rescale (bool): Rescale boundaries to the original image resolution
                if true, and keep the score_maps resolution if false.

        Returns:
            dict: A dict where boundary results are stored in
            ``boundary_result``.
        """

        assert check_argument.is_type_list(img_metas, dict)
        assert isinstance(rescale, bool)

        score_maps = score_maps.squeeze()
        boundaries = self.postprocessor(score_maps)

        if rescale:
            boundaries = self.resize_boundary(
                boundaries,
                1.0 / self.downsample_ratio / img_metas[0]['scale_factor'])

        results = dict(
            boundary_result=boundaries, filename=img_metas[0]['filename'])

        return results

    def loss(self, pred_maps, **kwargs):
        """Compute the loss for scene text detection.

        Args:
            pred_maps (Tensor): The input score maps of shape
                :math:`(NxCxHxW)`.

        Returns:
            dict: The dict for losses.
        """
        losses = self.loss_module(pred_maps, self.downsample_ratio, **kwargs)

        return losses
