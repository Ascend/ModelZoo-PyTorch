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
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines.formatting import DefaultFormatBundle

from mmocr.core.visualize import overlay_mask_img, show_feature


@PIPELINES.register_module()
class CustomFormatBundle(DefaultFormatBundle):
    """Custom formatting bundle.

    It formats common fields such as 'img' and 'proposals' as done in
    DefaultFormatBundle, while other fields such as 'gt_kernels' and
    'gt_effective_region_mask' will be formatted to DC as follows:

    - gt_kernels: to DataContainer (cpu_only=True)
    - gt_effective_mask: to DataContainer (cpu_only=True)

    Args:
        keys (list[str]): Fields to be formatted to DC only.
        call_super (bool): If True, format common fields
            by DefaultFormatBundle, else format fields in keys above only.
        visualize (dict): If flag=True, visualize gt mask for debugging.
    """

    def __init__(self,
                 keys=[],
                 call_super=True,
                 visualize=dict(flag=False, boundary_key=None)):

        super().__init__()
        self.visualize = visualize
        self.keys = keys
        self.call_super = call_super

    def __call__(self, results):

        if self.visualize['flag']:
            img = results['img'].astype(np.uint8)
            boundary_key = self.visualize['boundary_key']
            if boundary_key is not None:
                img = overlay_mask_img(img, results[boundary_key].masks[0])

            features = [img]
            names = ['img']
            to_uint8 = [1]

            for k in results['mask_fields']:
                for iter in range(len(results[k].masks)):
                    features.append(results[k].masks[iter])
                    names.append(k + str(iter))
                    to_uint8.append(0)
            show_feature(features, names, to_uint8)

        if self.call_super:
            results = super().__call__(results)

        for k in self.keys:
            results[k] = DC(results[k], cpu_only=True)

        return results

    def __repr__(self):
        return self.__class__.__name__
