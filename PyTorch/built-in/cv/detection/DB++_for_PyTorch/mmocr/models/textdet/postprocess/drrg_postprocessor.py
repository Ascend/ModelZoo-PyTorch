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
from mmocr.models.builder import POSTPROCESSOR
from .base_postprocessor import BasePostprocessor
from .utils import (clusters2labels, comps2boundaries, connected_components,
                    graph_propagation, remove_single)


@POSTPROCESSOR.register_module()
class DRRGPostprocessor(BasePostprocessor):
    """Merge text components and construct boundaries of text instances.

    Args:
        link_thr (float): The edge score threshold.
    """

    def __init__(self, link_thr, **kwargs):
        assert isinstance(link_thr, float)
        self.link_thr = link_thr

    def __call__(self, edges, scores, text_comps):
        """
        Args:
            edges (ndarray): The edge array of shape N * 2, each row is a node
                index pair that makes up an edge in graph.
            scores (ndarray): The edge score array of shape (N,).
            text_comps (ndarray): The text components.

        Returns:
            List[list[float]]: The predicted boundaries of text instances.
        """
        assert len(edges) == len(scores)
        assert text_comps.ndim == 2
        assert text_comps.shape[1] == 9

        vertices, score_dict = graph_propagation(edges, scores, text_comps)
        clusters = connected_components(vertices, score_dict, self.link_thr)
        pred_labels = clusters2labels(clusters, text_comps.shape[0])
        text_comps, pred_labels = remove_single(text_comps, pred_labels)
        boundaries = comps2boundaries(text_comps, pred_labels)

        return boundaries
