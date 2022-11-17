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
import os.path as osp

import mmcv
import numpy as np
import torch
from mmcv.image import tensor2imgs
from mmcv.parallel import DataContainer
from mmdet.core import encode_mask_results

from .utils import tensor2grayimgs


def retrieve_img_tensor_and_meta(data):
    """Retrieval img_tensor, img_metas and img_norm_cfg.

    Args:
        data (dict): One batch data from data_loader.

    Returns:
        tuple: Returns (img_tensor, img_metas, img_norm_cfg).

            - | img_tensor (Tensor): Input image tensor with shape
                :math:`(N, C, H, W)`.
            - | img_metas (list[dict]): The metadata of images.
            - | img_norm_cfg (dict): Config for image normalization.
    """

    if isinstance(data['img'], torch.Tensor):
        # for textrecog with batch_size > 1
        # and not use 'DefaultFormatBundle' in pipeline
        img_tensor = data['img']
        img_metas = data['img_metas'].data[0]
    elif isinstance(data['img'], list):
        if isinstance(data['img'][0], torch.Tensor):
            # for textrecog with aug_test and batch_size = 1
            img_tensor = data['img'][0]
        elif isinstance(data['img'][0], DataContainer):
            # for textdet with 'MultiScaleFlipAug'
            # and 'DefaultFormatBundle' in pipeline
            img_tensor = data['img'][0].data[0]
        img_metas = data['img_metas'][0].data[0]
    elif isinstance(data['img'], DataContainer):
        # for textrecog with 'DefaultFormatBundle' in pipeline
        img_tensor = data['img'].data[0]
        img_metas = data['img_metas'].data[0]

    must_keys = ['img_norm_cfg', 'ori_filename', 'img_shape', 'ori_shape']
    for key in must_keys:
        if key not in img_metas[0]:
            raise KeyError(
                f'Please add {key} to the "meta_keys" in the pipeline')

    img_norm_cfg = img_metas[0]['img_norm_cfg']
    if max(img_norm_cfg['mean']) <= 1:
        img_norm_cfg['mean'] = [255 * x for x in img_norm_cfg['mean']]
        img_norm_cfg['std'] = [255 * x for x in img_norm_cfg['std']]

    return img_tensor, img_metas, img_norm_cfg


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    is_kie=False,
                    show_score_thr=0.3):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        batch_size = len(result)
        if show or out_dir:
            if is_kie:
                img_tensor = data['img'].data[0]
                if img_tensor.shape[0] != 1:
                    raise KeyError('Visualizing KIE outputs in batches is'
                                   'currently not supported.')
                gt_bboxes = data['gt_bboxes'].data[0]
                img_metas = data['img_metas'].data[0]
                must_keys = ['img_norm_cfg', 'ori_filename', 'img_shape']
                for key in must_keys:
                    if key not in img_metas[0]:
                        raise KeyError(
                            f'Please add {key} to the "meta_keys" in config.')
                # for no visual model
                if np.prod(img_tensor.shape) == 0:
                    imgs = []
                    for img_meta in img_metas:
                        try:
                            img = mmcv.imread(img_meta['filename'])
                        except Exception as e:
                            print(f'Load image with error: {e}, '
                                  'use empty image instead.')
                            img = np.ones(
                                img_meta['img_shape'], dtype=np.uint8)
                        imgs.append(img)
                else:
                    imgs = tensor2imgs(img_tensor,
                                       **img_metas[0]['img_norm_cfg'])
                for i, img in enumerate(imgs):
                    h, w, _ = img_metas[i]['img_shape']
                    img_show = img[:h, :w, :]
                    if out_dir:
                        out_file = osp.join(out_dir,
                                            img_metas[i]['ori_filename'])
                    else:
                        out_file = None

                    model.module.show_result(
                        img_show,
                        result[i],
                        gt_bboxes[i],
                        show=show,
                        out_file=out_file)
            else:
                img_tensor, img_metas, img_norm_cfg = \
                    retrieve_img_tensor_and_meta(data)

                if img_tensor.size(1) == 1:
                    imgs = tensor2grayimgs(img_tensor, **img_norm_cfg)
                else:
                    imgs = tensor2imgs(img_tensor, **img_norm_cfg)
                assert len(imgs) == len(img_metas)

                for j, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                    img_shape, ori_shape = img_meta['img_shape'], img_meta[
                        'ori_shape']
                    img_show = img[:img_shape[0], :img_shape[1]]
                    img_show = mmcv.imresize(img_show,
                                             (ori_shape[1], ori_shape[0]))

                    if out_dir:
                        out_file = osp.join(out_dir, img_meta['ori_filename'])
                    else:
                        out_file = None

                    model.module.show_result(
                        img_show,
                        result[j],
                        show=show,
                        out_file=out_file,
                        score_thr=show_score_thr)

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
    return results
