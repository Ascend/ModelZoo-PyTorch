# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.nn as nn

from mmdet.core import bbox2result
from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector


@DETECTORS.register_module
class SingleStageInsDetector(BaseDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 mask_feat_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SingleStageInsDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        if mask_feat_head is not None:
            self.mask_feat_head = builder.build_head(mask_feat_head)

        self.bbox_head = builder.build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(SingleStageInsDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_mask_feat_head:
            if isinstance(self.mask_feat_head, nn.Sequential):
                for m in self.mask_feat_head:
                    m.init_weights()
            else:
                self.mask_feat_head.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        # diff
        img = img.npu()
        for i in range(len(gt_bboxes)):
            gt_bboxes[i] = gt_bboxes[i].npu()
        for i in range(len(gt_labels)):
            gt_labels[i] = gt_labels[i].npu()

        x = self.extract_feat(img)
        outs = self.bbox_head(x)

        if self.with_mask_feat_head:
            mask_feat_pred = self.mask_feat_head(
                x[self.mask_feat_head.
                      start_level:self.mask_feat_head.end_level + 1])
            loss_inputs = outs + (mask_feat_pred, gt_bboxes, gt_labels, gt_masks, img_metas, self.train_cfg)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, gt_masks, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_meta, rescale=False):
        # diff
        img = img.npu()
        x = self.extract_feat(img)
        outs = self.bbox_head(x, eval=True)
        if self.with_mask_feat_head:
            mask_feat_pred = self.mask_feat_head(
                x[self.mask_feat_head.
                      start_level:self.mask_feat_head.end_level + 1])
            seg_inputs = outs + (mask_feat_pred, img_meta, self.test_cfg, rescale)
        else:
            seg_inputs = outs + (img_meta, self.test_cfg, rescale)

        seg_result = self.bbox_head.get_seg(*seg_inputs)
        return seg_result

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError
