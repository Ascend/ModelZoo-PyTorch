"""
This file contains specific functions for computing losses on the RetinaNet
file
"""

import torch

from maskrcnn_benchmark.layers import SmoothL1Loss
from maskrcnn_benchmark.layers import AdjustSmoothL1Loss
from maskrcnn_benchmark.layers import SigmoidFocalLoss
from maskrcnn_benchmark.modeling.matcher import Matcher


class RetinaNetLossComputation(object):
    """
    This class computes the RetinaNet loss.
    """

    def __init__(self, cfg, proposal_matcher, box_coder):
        """
        Arguments:
            proposal_matcher (Matcher)
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher
        self.box_coder = box_coder
        self.num_classes = cfg.RETINANET.NUM_CLASSES - 1
        self.box_cls_loss_func = SigmoidFocalLoss(
            self.num_classes,
            cfg.RETINANET.LOSS_GAMMA,
            cfg.RETINANET.LOSS_ALPHA
        )
        if cfg.RETINANET.SELFADJUST_SMOOTH_L1:
            self.regression_loss = AdjustSmoothL1Loss(
                4,
                beta=cfg.RETINANET.BBOX_REG_BETA
            )
        else:
            self.regression_loss = SmoothL1Loss(
                beta=cfg.RETINANET.BBOX_REG_BETA
            )

    def _iou(self, boxlist1, anchors):

        area1 = boxlist1.area()
        area2 = (anchors[:, 2] - anchors[:, 0] + 1) * (anchors[:, 3] - anchors[:, 1] + 1)

        box1, box2 = boxlist1.bbox, anchors

        lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
        rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

        TO_REMOVE = 1

        wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        iou = inter / (area1[:, None] + area2 - inter)
        return iou

    def match_targets_to_anchors(self, anchor, target):
        match_quality_matrix = self._iou(target, anchor)

        matched_idxs = self.proposal_matcher(match_quality_matrix)

        # RPN doesn't need any fields from target
        # for creating the labels, so clear them all
        target = target.copy_with_fields(['labels'])
        # get the targets corresponding GT for each anchor
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, anchors_per_img, targets):
        labels = []
        regression_targets = []
        for anchors_per_image, targets_per_image in zip(anchors_per_img, targets):
            matched_targets = self.match_targets_to_anchors(
                anchors_per_image, targets_per_image
            )

            matched_idxs = matched_targets.get_field("matched_idxs")
            labels_per_image = matched_targets.get_field("labels").clone()

            # Background (negative examples)
            bg_indices = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_indices] = 0

            # discard indices that are between thresholds 
            # -1 will be ignored in SigmoidFocalLoss
            inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[inds_to_discard] = -1

            labels_per_image = labels_per_image.to(dtype=torch.float16)
            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, anchors_per_image
            )
            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)

        return labels, regression_targets

    def __call__(self, anchors_per_img, box_cls, box_regression, targets, C):
        """
        Arguments:
            anchors (list[BoxList])
            objectness (list[Tensor])
            box_regression (list[Tensor])
            targets (list[BoxList])

        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor
        """
        labels, regression_targets = self.prepare_targets(anchors_per_img, targets)
        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)
        pos_inds = labels > 0
        N = box_cls.size(0)
        box_cls = box_cls.reshape(-1, C)
        box_regression = box_regression.reshape(-1, 4)
        pos_inds_int = pos_inds.to(dtype=torch.float16).reshape(-1, 1)
        box_regression = box_regression * pos_inds_int
        regression_targets = regression_targets * pos_inds_int
        pos_cnt = pos_inds.sum()

        retinanet_regression_loss = self.regression_loss(
            box_regression,
            regression_targets,
            size_average=False,
        ) / (pos_cnt * 4)
        labels = labels.int()

        retinanet_cls_loss = self.box_cls_loss_func(
            box_cls,
            labels
        ) / (pos_cnt + N)
        return retinanet_cls_loss, retinanet_regression_loss


def make_retinanet_loss_evaluator(cfg, box_coder):
    matcher = Matcher(
        cfg.MODEL.RPN.FG_IOU_THRESHOLD,
        cfg.MODEL.RPN.BG_IOU_THRESHOLD,
        allow_low_quality_matches=cfg.RETINANET.LOW_QUALITY_MATCHES,
        low_quality_threshold=cfg.RETINANET.LOW_QUALITY_THRESHOLD
    )

    loss_evaluator = RetinaNetLossComputation(
        cfg, matcher, box_coder
    )
    return loss_evaluator
