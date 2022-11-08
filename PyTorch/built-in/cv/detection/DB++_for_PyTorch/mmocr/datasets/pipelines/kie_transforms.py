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
from mmcv import rescale_size
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines.formatting import DefaultFormatBundle, to_tensor


@PIPELINES.register_module()
class ResizeNoImg:
    """Image resizing without img.

    Used for KIE.
    """

    def __init__(self, img_scale, keep_ratio=True):
        self.img_scale = img_scale
        self.keep_ratio = keep_ratio

    def __call__(self, results):
        w, h = results['img_info']['width'], results['img_info']['height']
        if self.keep_ratio:
            (new_w, new_h) = rescale_size((w, h),
                                          self.img_scale,
                                          return_scale=False)
            w_scale = new_w / w
            h_scale = new_h / h
        else:
            (new_w, new_h) = self.img_scale

        w_scale = new_w / w
        h_scale = new_h / h
        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                dtype=np.float32)
        results['img_shape'] = (new_h, new_w, 1)
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = True

        return results


@PIPELINES.register_module()
class KIEFormatBundle(DefaultFormatBundle):
    """Key information extraction formatting bundle.

    Based on the DefaultFormatBundle, itt simplifies the pipeline of formatting
    common fields, including "img", "proposals", "gt_bboxes", "gt_labels",
    "gt_masks", "gt_semantic_seg", "relations" and "texts".
    These fields are formatted as follows.

    - img: (1) transpose, (2) to tensor, (3) to DataContainer (stack=True)
    - proposals: (1) to tensor, (2) to DataContainer
    - gt_bboxes: (1) to tensor, (2) to DataContainer
    - gt_bboxes_ignore: (1) to tensor, (2) to DataContainer
    - gt_labels: (1) to tensor, (2) to DataContainer
    - gt_masks: (1) to tensor, (2) to DataContainer (cpu_only=True)
    - gt_semantic_seg: (1) unsqueeze dim-0 (2) to tensor,
                       (3) to DataContainer (stack=True)
    - relations: (1) scale, (2) to tensor, (3) to DataContainer
    - texts: (1) to tensor, (2) to DataContainer
    """

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        super().__call__(results)
        if 'ann_info' in results:
            for key in ['relations', 'texts']:
                value = results['ann_info'][key]
                if key == 'relations' and 'scale_factor' in results:
                    scale_factor = results['scale_factor']
                    if isinstance(scale_factor, float):
                        sx = sy = scale_factor
                    else:
                        sx, sy = results['scale_factor'][:2]
                    r = sx / sy
                    factor = np.array([sx, sy, r, 1, r]).astype(np.float32)
                    value = value * factor[None, None]
                results[key] = DC(to_tensor(value))
        return results

    def __repr__(self):
        return self.__class__.__name__
