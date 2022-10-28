# Copyright 2022 Huawei Technologies Co., Ltd.
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
# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.runner import force_fp32

from mmdet.core import (build_assigner, build_bbox_coder,
                        build_prior_generator, build_sampler, multi_apply)
from ..builder import HEADS
from ..losses import smooth_l1_loss
from .anchor_head import AnchorHead
from ...utils import set_index


@HEADS.register_module()
class AscendSSDHead(AnchorHead):
    """SSD head used in https://arxiv.org/abs/1512.02325.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        stacked_convs (int): Number of conv layers in cls and reg tower.
            Default: 0.
        feat_channels (int): Number of hidden channels when stacked_convs
            > 0. Default: 256.
        use_depthwise (bool): Whether to use DepthwiseSeparableConv.
            Default: False.
        conv_cfg (dict): Dictionary to construct and config conv layer.
            Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: None.
        act_cfg (dict): Dictionary to construct and config activation layer.
            Default: None.
        anchor_generator (dict): Config dict for anchor generator
        bbox_coder (dict): Config of bounding box coder.
        reg_decoded_bbox (bool): If true, the regression loss would be
            applied directly on decoded bounding boxes, converting both
            the predicted boxes and regression targets to absolute
            coordinates format. Default False. It should be `True` when
            using `IoULoss`, `GIoULoss`, or `DIoULoss` in the bbox head.
        train_cfg (dict): Training config of anchor head.
        test_cfg (dict): Testing config of anchor head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """  # noqa: W605

    def __init__(self,
                 num_classes=80,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 stacked_convs=0,
                 feat_channels=256,
                 use_depthwise=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 anchor_generator=dict(
                     type='SSDAnchorGenerator',
                     scale_major=False,
                     input_size=300,
                     strides=[8, 16, 32, 64, 100, 300],
                     ratios=([2], [2, 3], [2, 3], [2, 3], [2], [2]),
                     basesize_ratio_range=(0.1, 0.9)),
                 bbox_coder=dict(
                     type='DeltaXYWHBBoxCoder',
                     clip_border=True,
                     target_means=[.0, .0, .0, .0],
                     target_stds=[1.0, 1.0, 1.0, 1.0],
                 ),
                 reg_decoded_bbox=False,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=dict(
                     type='Xavier',
                     layer='Conv2d',
                     distribution='uniform',
                     bias=0)):
        super(AnchorHead, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.stacked_convs = stacked_convs
        self.feat_channels = feat_channels
        self.use_depthwise = use_depthwise
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.cls_out_channels = num_classes + 1  # add background class
        self.prior_generator = build_prior_generator(anchor_generator)

        # Usually the numbers of anchors for each level are the same
        # except SSD detectors. So it is an int in the most dense
        # heads but a list of int in SSDHead
        self.num_base_priors = self.prior_generator.num_base_priors

        self._init_layers()

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.reg_decoded_bbox = reg_decoded_bbox
        self.use_sigmoid_cls = False
        self.cls_focal_loss = False
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        # set sampling=False for archor_target
        self.sampling = False
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # SSD sampling=False so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.fp16_enabled = False

        self.concat_anchors = None
        self.concat_valid_flags = None
        self.max_gt_labels = (32, 64, 128)
        self.concat_gt_bboxes = {key: None for key in self.max_gt_labels}

    @property
    def num_anchors(self):
        """
        Returns:
            list[int]: Number of base_anchors on each point of each level.
        """
        warnings.warn('DeprecationWarning: `num_anchors` is deprecated, '
                      'please use "num_base_priors" instead')
        return self.num_base_priors

    def _init_layers(self):
        """Initialize layers of the head."""
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        conv = DepthwiseSeparableConvModule \
            if self.use_depthwise else ConvModule

        for channel, num_base_priors in zip(self.in_channels,
                                            self.num_base_priors):
            cls_layers = []
            reg_layers = []
            in_channel = channel
            # build stacked conv tower, not used in default ssd
            for i in range(self.stacked_convs):
                cls_layers.append(
                    conv(
                        in_channel,
                        self.feat_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                reg_layers.append(
                    conv(
                        in_channel,
                        self.feat_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                in_channel = self.feat_channels
            # SSD-Lite head
            if self.use_depthwise:
                cls_layers.append(
                    ConvModule(
                        in_channel,
                        in_channel,
                        3,
                        padding=1,
                        groups=in_channel,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                reg_layers.append(
                    ConvModule(
                        in_channel,
                        in_channel,
                        3,
                        padding=1,
                        groups=in_channel,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
            cls_layers.append(
                nn.Conv2d(
                    in_channel,
                    num_base_priors * self.cls_out_channels,
                    kernel_size=1 if self.use_depthwise else 3,
                    padding=0 if self.use_depthwise else 1))
            reg_layers.append(
                nn.Conv2d(
                    in_channel,
                    num_base_priors * 4,
                    kernel_size=1 if self.use_depthwise else 3,
                    padding=0 if self.use_depthwise else 1))
            self.cls_convs.append(nn.Sequential(*cls_layers))
            self.reg_convs.append(nn.Sequential(*reg_layers))

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:
                cls_scores (list[Tensor]): Classification scores for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_anchors * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_anchors * 4.
        """
        cls_scores = []
        bbox_preds = []
        for feat, reg_conv, cls_conv in zip(feats, self.reg_convs,
                                            self.cls_convs):
            cls_scores.append(cls_conv(feat))
            bbox_preds.append(reg_conv(feat))
        return cls_scores, bbox_preds

    # gt_bboxes batch
    def get_concat_gt_bboxes(self, gt_bboxes_list, num_images, gt_nums, device, max_gt_labels):
        if gt_bboxes_list is None:
            concat_gt_bboxes = None
        else:
            if self.concat_gt_bboxes.get(max_gt_labels) is None:
                concat_gt_bboxes = torch.zeros((num_images, max_gt_labels, 4),
                                               dtype=gt_bboxes_list[0].dtype,
                                               device=device)
                concat_anchor = self.concat_anchors
                min_anchor_num = torch.min(concat_anchor.view(-1))
                concat_gt_bboxes[:, :, :2] = min_anchor_num - 10
                concat_gt_bboxes[:, :, 2:] = min_anchor_num - 5
                self.concat_gt_bboxes[max_gt_labels] = concat_gt_bboxes.clone()
            else:
                concat_gt_bboxes = self.concat_gt_bboxes.get(max_gt_labels).clone()
            for index_imgs, gt_bboxes in enumerate(gt_bboxes_list):
                concat_gt_bboxes[index_imgs, :gt_nums[index_imgs]] = gt_bboxes
        return concat_gt_bboxes

    # gt_labels batch
    def get_concat_gt_labels(self, gt_labels_list, num_images, gt_nums, device, max_gt_labels):
        if gt_labels_list is None:
            concat_gt_labels = None
        else:
            concat_gt_labels = torch.zeros((num_images, max_gt_labels),
                                           dtype=gt_labels_list[0].dtype,
                                           device=device)
            for index_imgs, gt_labels in enumerate(gt_labels_list):
                concat_gt_labels[index_imgs, :gt_nums[index_imgs]] = gt_labels

        return concat_gt_labels

    # gt_bboxes_ignore batch
    def get_concat_gt_bboxes_ignore(self, gt_bboxes_ignore_list, num_images, gt_nums, device):
        if gt_bboxes_ignore_list is None:
            concat_gt_bboxes_ignore = None
        else:
            raise RuntimeError("gt_bboxes_ignore not support yet")
        return concat_gt_bboxes_ignore

    # get concat_anchors
    def get_concat_anchors(self, featmap_sizes, img_metas, device="npu"):
        concat_anchors = self.concat_anchors
        concat_valid_flags = self.concat_valid_flags
        if concat_anchors is None or concat_valid_flags is None:
            anchors_list, valid_flags_list = self.get_anchors(featmap_sizes, img_metas, device)
            num_imgs = len(img_metas)
            assert len(anchors_list) == len(valid_flags_list) == num_imgs
            concat_anchors_list = []
            concat_valid_flags_list = []
            for index_images in range(num_imgs):
                assert len(anchors_list[index_images]) == len(valid_flags_list[index_images])
                concat_anchors_list.append(torch.cat(anchors_list[index_images]))
                concat_valid_flags_list.append(torch.cat(valid_flags_list[index_images]))
            concat_anchors = torch.cat([torch.unsqueeze(anchor, 0) for anchor in concat_anchors_list], 0)
            concat_valid_flags = torch.cat([torch.unsqueeze(concat_valid_flags, 0)
                                            for concat_valid_flags in concat_valid_flags_list],
                                           0)
            self.concat_anchors = concat_anchors
            self.concat_valid_flags = concat_valid_flags
        return concat_anchors, concat_valid_flags

    def _get_concat_targets(self,
                            concat_anchors,
                            concat_valid_flags,
                            concat_gt_bboxes,
                            concat_gt_bboxes_ignore,
                            concat_gt_labels,
                            img_metas,
                            label_channels=1,
                            unmap_outputs=True):
        num_imgs, num_anchors, _ = concat_anchors.size()
        # assign gt and sample concat_anchors
        assign_result = self.assigner.assign(
            concat_anchors, concat_gt_bboxes, concat_gt_bboxes_ignore,
            None if self.sampling else concat_gt_labels)
        sampling_result = None
        concat_pos_mask = assign_result.concat_pos_mask
        concat_neg_mask = assign_result.concat_neg_mask
        concat_anchor_gt_indes = assign_result.concat_anchor_gt_indes
        concat_anchor_gt_labels = assign_result.concat_anchor_gt_labels

        concat_anchor_gt_bboxes = torch.zeros(concat_anchors.size(),
                                              dtype=concat_anchors.dtype,
                                              device=concat_anchors.device)
        for index_imgs in range(num_imgs):
            concat_anchor_gt_bboxes[index_imgs] = torch.index_select(concat_gt_bboxes[index_imgs], 0,
                                                                     concat_anchor_gt_indes[index_imgs])

        concat_bbox_targets = torch.zeros_like(concat_anchors)
        concat_bbox_weights = torch.zeros_like(concat_anchors)
        concat_labels = concat_anchors.new_full((num_imgs, num_anchors), self.num_classes, dtype=torch.int)
        concat_label_weights = concat_anchors.new_zeros((num_imgs, num_anchors), dtype=torch.float)

        if not self.reg_decoded_bbox:
            concat_pos_bbox_targets = self.bbox_coder.encode(concat_anchors, concat_anchor_gt_bboxes)
        else:
            concat_pos_bbox_targets = concat_anchor_gt_bboxes
        concat_bbox_targets = set_index(concat_bbox_targets, concat_pos_mask, concat_pos_bbox_targets)
        concat_bbox_weights = set_index(concat_bbox_weights, concat_pos_mask, 1.0)
        if concat_gt_labels is None:
            concat_labels = set_index(concat_labels, concat_pos_mask, 0.0)
        else:
            concat_labels = set_index(concat_labels, concat_pos_mask, concat_anchor_gt_labels)
        if self.train_cfg.pos_weight <= 0:
            concat_label_weights = set_index(concat_label_weights, concat_pos_mask, 1.0)
        else:
            concat_label_weights = set_index(concat_label_weights, concat_pos_mask, self.train_cfg.pos_weight)
        concat_label_weights = set_index(concat_label_weights, concat_neg_mask, 1.0)
        return (concat_labels, concat_label_weights, concat_bbox_targets, concat_bbox_weights, concat_pos_mask,
                concat_neg_mask, sampling_result)

    # concat target
    def get_concat_targets(self,
                           concat_anchors,
                           concat_valid_flags,
                           concat_gt_bboxes,
                           img_metas,
                           concat_gt_bboxes_ignore=None,
                           concat_gt_labels=None,
                           label_channels=1,
                           unmap_outputs=True,
                           return_sampling_results=False):
        # compute targets for each image
        num_imgs = len(img_metas)
        results = self._get_concat_targets(
            concat_anchors,
            concat_valid_flags,
            concat_gt_bboxes,
            concat_gt_bboxes_ignore,
            concat_gt_labels,
            img_metas,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs)

        (concat_labels, concat_label_weights, concat_bbox_targets, concat_bbox_weights, concat_pos_mask,
         concat_neg_mask, sampling_result) = results[:7]
        rest_results = list(results[7:])  # user-added return values
        # no valid anchors
        if concat_labels is None:
            return None
        # sampled anchors of all images
        min_num = torch.ones((num_imgs, ), dtype=concat_pos_mask.dtype, device=concat_pos_mask.device)
        num_total_pos = torch.sum(torch.max(torch.sum(concat_pos_mask, dim=1), min_num))
        num_total_neg = torch.sum(torch.max(torch.sum(concat_neg_mask, dim=1), min_num))
        return results + (num_total_pos, num_total_neg)

    def concat_loss(self,
                    concat_cls_score, concat_bbox_pred,
                    concat_anchor, concat_labels,
                    concat_label_weights,
                    concat_bbox_targets, concat_bbox_weights,
                    concat_pos_mask, concat_neg_mask,
                    num_total_samples):
        num_images, num_anchors, _ = concat_anchor.size()

        concat_loss_cls_all = F.cross_entropy(
            concat_cls_score.view((-1, self.cls_out_channels)), concat_labels.view(-1),
            reduction='none').view(concat_label_weights.size()) * concat_label_weights
        # # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        concat_num_pos_samples = torch.sum(concat_pos_mask, dim=1)
        concat_num_neg_samples = self.train_cfg.neg_pos_ratio * concat_num_pos_samples

        concat_num_neg_samples_max = torch.sum(concat_neg_mask, dim=1)
        concat_num_neg_samples = torch.min(concat_num_neg_samples, concat_num_neg_samples_max)

        concat_topk_loss_cls_neg, _ = torch.topk(concat_loss_cls_all * concat_neg_mask, k=num_anchors, dim=1)
        concat_loss_cls_pos = torch.sum(concat_loss_cls_all * concat_pos_mask, dim=1)

        anchor_index = torch.arange(end=num_anchors, dtype=torch.float, device=concat_anchor.device).view((1, -1))
        topk_loss_neg_mask = (anchor_index < concat_num_neg_samples.view(-1, 1)).float()

        concat_loss_cls_neg = torch.sum(concat_topk_loss_cls_neg * topk_loss_neg_mask, dim=1)
        loss_cls = (concat_loss_cls_pos + concat_loss_cls_neg) / num_total_samples

        if self.reg_decoded_bbox:
            raise RuntimeError

        loss_bbox = smooth_l1_loss(
            concat_bbox_pred,
            concat_bbox_targets,
            concat_bbox_weights,
            reduction="mean",
            beta=self.train_cfg.smoothl1_beta,
            avg_factor=num_total_samples)
        return loss_cls[None], loss_bbox

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device
        num_images = len(img_metas)
        gt_nums = [len(gt_bbox) for gt_bbox in gt_bboxes]
        gt_nums_max = max(gt_nums)
        max_gt_labels = self.max_gt_labels[0]
        for max_gt_labels in self.max_gt_labels:
            if max_gt_labels >= gt_nums_max:
                break
        # concat anchors
        concat_anchors, concat_valid_flags = self.get_concat_anchors(featmap_sizes, img_metas, device=device)
        # concat gt_bboxes  gt_labels gt_bboxes_ignore
        concat_gt_bboxes = self.get_concat_gt_bboxes(gt_bboxes, num_images, gt_nums, device, max_gt_labels)
        concat_gt_labels = self.get_concat_gt_labels(gt_labels, num_images, gt_nums, device, max_gt_labels)
        concat_gt_bboxes_ignore = self.get_concat_gt_bboxes_ignore(gt_bboxes_ignore, num_images, gt_nums, device)
        #
        cls_reg_targets = self.get_concat_targets(
            concat_anchors,
            concat_valid_flags,
            concat_gt_bboxes,
            img_metas,
            concat_gt_bboxes_ignore=concat_gt_bboxes_ignore,
            concat_gt_labels=concat_gt_labels,
            label_channels=1,
            unmap_outputs=True)

        if cls_reg_targets is None:
            return None
        (concat_labels, concat_label_weights, concat_bbox_targets, concat_bbox_weights, concat_pos_mask,
         concat_neg_mask, sampling_result, num_total_pos, num_total_neg) = cls_reg_targets

        num_imgs = len(img_metas)
        concat_cls_score = torch.cat([
            s.permute(0, 2, 3, 1).reshape(
                num_imgs, -1, self.cls_out_channels) for s in cls_scores
        ], 1)

        concat_bbox_pred = torch.cat([
            b.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for b in bbox_preds
        ], -2)

        # concat all level anchors to a single tensor
        concat_anchor = self.concat_anchors

        concat_losses_cls, concat_losses_bbox = self.concat_loss(
            concat_cls_score, concat_bbox_pred,
            concat_anchor, concat_labels,
            concat_label_weights,
            concat_bbox_targets, concat_bbox_weights,
            concat_pos_mask, concat_neg_mask,
            num_total_pos)
        losses_cls = [concat_losses_cls[:, index_imgs] for index_imgs in range(num_imgs)]
        losses_bbox = [losses_bbox for losses_bbox in concat_losses_bbox]
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)